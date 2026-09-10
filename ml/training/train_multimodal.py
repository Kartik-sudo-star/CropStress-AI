"""
Training script for Multimodal Deepfake Detector
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

from ml.models.multimodal_detector import MultimodalDeepfakeDetector, create_multimodal_detector, save_multimodal_detector
from ml.preprocessing.video_preprocessing import VideoPreprocessor, create_video_preprocessor_from_config
from ml.preprocessing.audio_preprocessing import AudioPreprocessor, create_audio_preprocessor_from_config


class MultimodalDeepfakeDataset(torch.utils.data.Dataset):
    """Dataset for multimodal deepfake detection (video + audio)."""
    
    def __init__(
        self,
        manifest_path: str,
        video_preprocessor: VideoPreprocessor,
        audio_preprocessor: AudioPreprocessor,
        is_training: bool = True,
        use_face_detection: bool = True
    ):
        import pandas as pd
        self.df = pd.read_csv(manifest_path)
        self.video_preprocessor = video_preprocessor
        self.audio_preprocessor = audio_preprocessor
        self.is_training = is_training
        self.use_face_detection = use_face_detection
        
        # Need video_path, audio_path, label
        required = ["video_path", "audio_path", "label"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, "0": 0, "1": 1}
        self.df["label_idx"] = self.df["label"].map(self.label_map)
        self.df = self.df.dropna(subset=["label_idx"])
        self.df["label_idx"] = self.df["label_idx"].astype(int)
        
        logger.info(f"Loaded {len(self.df)} multimodal samples from {manifest_path}")
        logger.info(f"Class distribution: {self.df['label_idx'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        audio_path = row["audio_path"]
        label = row["label_idx"]
        
        try:
            video_tensor, _, _ = self.video_preprocessor.preprocess(
                video_path,
                is_training=self.is_training,
                use_face_detection=self.use_face_detection
            )
        except Exception as e:
            logger.warning(f"Failed to process video {video_path}: {e}")
            T = self.video_preprocessor.frame_sampler.num_frames
            C, H, W = 3, *self.video_preprocessor.target_size
            video_tensor = torch.zeros(T, C, H, W)
        
        try:
            waveform, mel_spec = self.audio_preprocessor.preprocess(
                audio_path,
                is_training=self.is_training,
                return_waveform=True
            )
        except Exception as e:
            logger.warning(f"Failed to process audio {audio_path}: {e}")
            waveform = torch.zeros(1, self.audio_preprocessor.segment_samples)
            mel_spec = torch.zeros(1, self.audio_preprocessor.spectrogram_config.get("n_mels", 80), self.audio_preprocessor.target_length)
        
        return video_tensor, waveform, label


def create_data_loaders(config, video_preprocessor, audio_preprocessor, batch_size, num_workers=4):
    paths = config["paths"]
    
    train_dataset = MultimodalDeepfakeDataset(
        os.path.join(paths["data_train"], "manifest.csv"),
        video_preprocessor,
        audio_preprocessor,
        is_training=True,
        use_face_detection=config["dataset"]["video"]["face_detection"]["enabled"]
    )
    
    val_dataset = MultimodalDeepfakeDataset(
        os.path.join(paths["data_validation"], "manifest.csv"),
        video_preprocessor,
        audio_preprocessor,
        is_training=False,
        use_face_detection=config["dataset"]["video"]["face_detection"]["enabled"]
    )
    
    test_dataset = MultimodalDeepfakeDataset(
        os.path.join(paths["data_test"], "manifest.csv"),
        video_preprocessor,
        audio_preprocessor,
        is_training=False,
        use_face_detection=config["dataset"]["video"]["face_detection"]["enabled"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, mixed_precision=False, gradient_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (video, audio, labels) in enumerate(pbar):
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with autocast():
                outputs = model(video, audio)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(video, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}", "acc": f"{100.*correct/total:.2f}%"})
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, mixed_precision=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for video, audio, labels in tqdm(loader, desc="Validation"):
            video = video.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if mixed_precision:
                with autocast():
                    outputs = model(video, audio)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(video, audio)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
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
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Deepfake Detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.epochs:
        config["multimodal_detector"]["training"]["epochs"] = args.epochs
    
    device = torch.device(config["general"]["device"])
    logger.add(config["paths"]["logs_dir"] + "/train_multimodal.log", rotation="10 MB")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config["general"]["random_seed"])
    import numpy as np
    np.random.seed(config["general"]["random_seed"])
    
    # Create preprocessors
    video_preprocessor = create_video_preprocessor_from_config(config)
    audio_preprocessor = create_audio_preprocessor_from_config(config)
    
    video_preprocessor.save(os.path.join(config["paths"]["models_dir"], "video_detector", "preprocessor.pkl"))
    audio_preprocessor.save(os.path.join(config["paths"]["models_dir"], "audio_detector", "preprocessor.pkl"))
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        video_preprocessor,
        audio_preprocessor,
        config["multimodal_detector"]["training"]["batch_size"],
        config["general"]["num_workers"]
    )
    
    # Create model
    model = create_multimodal_detector(config).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())}")
    
    # Loss
    training_cfg = config["multimodal_detector"]["training"]
    if training_cfg["loss_function"] == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=training_cfg["loss_params"].get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if training_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])
    
    # Scheduler
    if training_cfg["scheduler"] == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["scheduler_params"]["T_max"], eta_min=training_cfg["scheduler_params"]["eta_min"])
    elif training_cfg["scheduler"] == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=training_cfg["scheduler_params"]["factor"], patience=training_cfg["scheduler_params"]["patience"], min_lr=training_cfg["scheduler_params"]["min_lr"])
    else:
        scheduler = None
    
    scaler = GradScaler() if training_cfg.get("mixed_precision", False) else None
    
    start_epoch = 0
    best_val_auc = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint.get("metrics", {}).get("val_auc", 0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    epochs = training_cfg["epochs"]
    early_stopping_patience = training_cfg["early_stopping"]["patience"]
    early_stopping_counter = 0
    
    model_dir = Path(config["paths"]["models_dir"]) / "multimodal_detector"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, training_cfg.get("mixed_precision", False), training_cfg.get("gradient_clip", 1.0))
        
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device, training_cfg.get("mixed_precision", False))
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_auc = val_metrics["roc_auc"]
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
            save_multimodal_detector(model, str(model_dir / "best_model.pth"), optimizer=optimizer, epoch=epoch, metrics={"val_auc": val_auc, **val_metrics}, config=config)
            logger.info(f"Saved best model (AUC: {val_auc:.4f})")
        else:
            early_stopping_counter += 1
        
        save_multimodal_detector(model, str(model_dir / f"checkpoint_epoch_{epoch}.pth"), optimizer=optimizer, epoch=epoch, metrics={"val_auc": val_auc, **val_metrics}, config=config)
        
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered")
            break
    
    # Final test
    logger.info("\n=== Final Test Evaluation ===")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logger.info(f"Test Metrics: {test_metrics}")
    
    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()