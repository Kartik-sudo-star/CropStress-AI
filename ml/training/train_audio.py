"""
Training script for Audio Deepfake Detector
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from loguru import logger
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.audio_detector import RawNet2, AMSoftmax, create_audio_detector, save_audio_detector
from ml.preprocessing.audio_preprocessing import AudioPreprocessor, create_audio_preprocessor_from_config


class DeepfakeAudioDataset(torch.utils.data.Dataset):
    """Dataset for audio deepfake detection."""
    
    def __init__(
        self,
        manifest_path: str,
        preprocessor: AudioPreprocessor,
        is_training: bool = True,
        return_waveform: bool = True  # RawNet2 needs waveform
    ):
        import pandas as pd
        self.df = pd.read_csv(manifest_path)
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.return_waveform = return_waveform
        
        required = ["audio_path", "label"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.label_map = {"real": 0, "fake": 1, "bonafide": 0, "spoof": 1, "Real": 0, "Fake": 1, "0": 0, "1": 1}
        self.df["label_idx"] = self.df["label"].map(self.label_map)
        self.df = self.df.dropna(subset=["label_idx"])
        self.df["label_idx"] = self.df["label_idx"].astype(int)
        
        logger.info(f"Loaded {len(self.df)} audio samples from {manifest_path}")
        logger.info(f"Class distribution: {self.df['label_idx'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = row["label_idx"]
        
        try:
            if self.return_waveform:
                waveform, mel_spec = self.preprocessor.preprocess(
                    audio_path,
                    is_training=self.is_training,
                    return_waveform=True
                )
                return waveform, mel_spec, label
            else:
                mel_spec = self.preprocessor.preprocess(
                    audio_path,
                    is_training=self.is_training,
                    return_waveform=False
                )
                return mel_spec, label
        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            if self.return_waveform:
                waveform = torch.zeros(1, self.preprocessor.segment_samples)
                mel_spec = torch.zeros(1, self.preprocessor.spectrogram_config.get("n_mels", 80), self.preprocessor.target_length)
                return waveform, mel_spec, label
            else:
                mel_spec = torch.zeros(1, self.preprocessor.spectrogram_config.get("n_mels", 80), self.preprocessor.target_length)
                return mel_spec, label


def create_data_loaders(config, preprocessor, batch_size, num_workers=4, return_waveform=True):
    paths = config["paths"]
    
    train_dataset = DeepfakeAudioDataset(
        os.path.join(paths["data_train"], "manifest.csv"),
        preprocessor,
        is_training=True,
        return_waveform=return_waveform
    )
    
    val_dataset = DeepfakeAudioDataset(
        os.path.join(paths["data_validation"], "manifest.csv"),
        preprocessor,
        is_training=False,
        return_waveform=return_waveform
    )
    
    test_dataset = DeepfakeAudioDataset(
        os.path.join(paths["data_test"], "manifest.csv"),
        preprocessor,
        is_training=False,
        return_waveform=return_waveform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, mixed_precision=False, gradient_clip=3.0, use_amsoftmax=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:
            waveform, mel_spec, labels = batch
            waveform = waveform.to(device, non_blocking=True)
            mel_spec = mel_spec.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            inputs = waveform  # RawNet2 uses waveform
        else:
            mel_spec, labels = batch
            mel_spec = mel_spec.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            inputs = mel_spec
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with autocast():
                if use_amsoftmax:
                    # AMSoftmax needs embeddings and labels
                    embeddings = model.get_embeddings(inputs)
                    outputs = criterion(embeddings, labels)
                    loss = outputs  # criterion returns loss directly
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_amsoftmax:
                embeddings = model.get_embeddings(inputs)
                loss = criterion(embeddings, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        
        if use_amsoftmax:
            # For AMSoftmax, compute accuracy from cosine similarity
            with torch.no_grad():
                embeddings = model.get_embeddings(inputs)
                weight = criterion.weight
                cos_theta = F.linear(F.normalize(embeddings), F.normalize(weight))
                _, predicted = cos_theta.max(1)
        else:
            _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}", "acc": f"{100.*correct/total:.2f}%"})
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, mixed_precision=False, use_amsoftmax=False):
    import torch.nn.functional as F
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if len(batch) == 3:
                waveform, mel_spec, labels = batch
                waveform = waveform.to(device, non_blocking=True)
                mel_spec = mel_spec.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inputs = waveform
            else:
                mel_spec, labels = batch
                mel_spec = mel_spec.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inputs = mel_spec
            
            if mixed_precision:
                with autocast():
                    if use_amsoftmax:
                        embeddings = model.get_embeddings(inputs)
                        loss = criterion(embeddings, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
            else:
                if use_amsoftmax:
                    embeddings = model.get_embeddings(inputs)
                    loss = criterion(embeddings, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            if use_amsoftmax:
                embeddings = model.get_embeddings(inputs)
                weight = criterion.weight
                cos_theta = F.linear(F.normalize(embeddings), F.normalize(weight))
                probs = torch.softmax(cos_theta * criterion.scale, dim=1)
                _, predicted = cos_theta.max(1)
            else:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def compute_metrics(labels, preds, probs):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    # EER (Equal Error Rate)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "eer": float(eer),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.epochs:
        config["audio_detector"]["training"]["epochs"] = args.epochs
    
    device = torch.device(config["general"]["device"])
    logger.add(config["paths"]["logs_dir"] + "/train_audio.log", rotation="10 MB")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config["general"]["random_seed"])
    import numpy as np
    np.random.seed(config["general"]["random_seed"])
    
    # Create preprocessor
    preprocessor = create_audio_preprocessor_from_config(config)
    preprocessor.save(os.path.join(config["paths"]["models_dir"], "audio_detector", "preprocessor.pkl"))
    
    # Determine if RawNet2 (needs waveform)
    architecture = config["audio_detector"]["architecture"]
    return_waveform = architecture in ["rawnet2", "wav2vec2"]
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, preprocessor,
        config["audio_detector"]["training"]["batch_size"],
        config["general"]["num_workers"],
        return_waveform=return_waveform
    )
    
    # Create model
    model = create_audio_detector(config).to(device)
    logger.info(f"Model: {architecture}, params: {sum(p.numel() for p in model.parameters())}")
    
    # Loss
    training_cfg = config["audio_detector"]["training"]
    use_amsoftmax = training_cfg["loss_function"] in ["amsoftmax", "aamsoftmax"]
    
    if use_amsoftmax:
        embedding_dim = config["audio_detector"]["embedding_dim"]
        criterion = AMSoftmax(
            embedding_dim,
            training_cfg["num_classes"],
            margin=training_cfg["loss_params"].get("margin", 0.2),
            scale=training_cfg["loss_params"].get("scale", 30)
        ).to(device)
    elif training_cfg["loss_function"] == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=training_cfg["loss_params"].get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if training_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])
    elif training_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=training_cfg["learning_rate"], momentum=0.9, weight_decay=training_cfg["weight_decay"])
    
    # Scheduler
    if training_cfg["scheduler"] == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=training_cfg["scheduler_params"]["factor"],
            patience=training_cfg["scheduler_params"]["patience"],
            min_lr=training_cfg["scheduler_params"]["min_lr"]
        )
    elif training_cfg["scheduler"] == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["scheduler_params"]["T_max"], eta_min=training_cfg["scheduler_params"]["eta_min"])
    else:
        scheduler = None
    
    scaler = GradScaler() if training_cfg.get("mixed_precision", False) else None
    
    start_epoch = 0
    best_val_eer = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_eer = checkpoint.get("metrics", {}).get("val_eer", float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    epochs = training_cfg["epochs"]
    early_stopping_patience = training_cfg["early_stopping"]["patience"]
    early_stopping_counter = 0
    
    model_dir = Path(config["paths"]["models_dir"]) / "audio_detector"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, training_cfg.get("mixed_precision", False),
            training_cfg.get("gradient_clip", 3.0),
            use_amsoftmax
        )
        
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device,
            training_cfg.get("mixed_precision", False),
            use_amsoftmax
        )
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_eer = val_metrics["eer"]
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val EER: {val_eer:.4f}")
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_eer)
            else:
                scheduler.step()
        
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            early_stopping_counter = 0
            save_audio_detector(model, str(model_dir / "best_model.pth"), optimizer=optimizer, epoch=epoch, metrics={"val_eer": val_eer, **val_metrics}, config=config)
            logger.info(f"Saved best model (EER: {val_eer:.4f})")
        else:
            early_stopping_counter += 1
        
        save_audio_detector(model, str(model_dir / f"checkpoint_epoch_{epoch}.pth"), optimizer=optimizer, epoch=epoch, metrics={"val_eer": val_eer, **val_metrics}, config=config)
        
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered")
            break
    
    # Final test
    logger.info("\n=== Final Test Evaluation ===")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(model, test_loader, criterion, device, use_amsoftmax=use_amsoftmax)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logger.info(f"Test Metrics: {test_metrics}")
    
    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()