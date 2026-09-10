"""
Training script for Image Deepfake Detector
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from loguru import logger
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.image_detector import DeepfakeImageDetector, create_image_detector, save_image_detector, load_image_detector
from ml.preprocessing.image_preprocessing import ImagePreprocessor, create_preprocessor_from_config
from ml.explainability.gradcam import GradCAM


class DeepfakeImageDataset(torch.utils.data.Dataset):
    """Dataset for image deepfake detection."""
    
    def __init__(
        self,
        manifest_path: str,
        preprocessor: ImagePreprocessor,
        is_training: bool = True,
        use_face_detection: bool = True
    ):
        import pandas as pd
        self.df = pd.read_csv(manifest_path)
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.use_face_detection = use_face_detection
        
        # Validate required columns
        required = ["image_path", "label"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Map string labels to integers
        self.label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, "0": 0, "1": 1}
        self.df["label_idx"] = self.df["label"].map(self.label_map)
        
        # Keep only rows that carry an image path (manifests are shared
        # across modalities: image / video / audio rows coexist)
        self.df = self.df.dropna(subset=["image_path"])
        self.df = self.df[self.df["image_path"].astype(str).str.len() > 0]

        # Remove invalid labels
        self.df = self.df.dropna(subset=["label_idx"])
        self.df["label_idx"] = self.df["label_idx"].astype(int)
        
        logger.info(f"Loaded {len(self.df)} samples from {manifest_path}")
        logger.info(f"Class distribution: {self.df['label_idx'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label_idx"]
        
        # Load and preprocess image
        try:
            tensor = self.preprocessor.preprocess_file(
                image_path,
                is_training=self.is_training,
                use_face_detection=self.use_face_detection
            )
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            # Return dummy tensor
            tensor = torch.zeros(3, *self.preprocessor.target_size)
        
        return tensor, label


def create_data_loaders(
    config: dict,
    preprocessor: ImagePreprocessor,
    batch_size: int,
    num_workers: int = 4
):
    """Create train/val/test data loaders."""
    
    paths = config["paths"]
    
    # Training dataset
    train_dataset = DeepfakeImageDataset(
        os.path.join(paths["data_train"], "manifest.csv"),
        preprocessor,
        is_training=True,
        use_face_detection=config["dataset"]["image"]["face_detection"]["enabled"]
    )
    
    # Validation dataset
    val_dataset = DeepfakeImageDataset(
        os.path.join(paths["data_validation"], "manifest.csv"),
        preprocessor,
        is_training=False,
        use_face_detection=config["dataset"]["image"]["face_detection"]["enabled"]
    )
    
    # Test dataset
    test_dataset = DeepfakeImageDataset(
        os.path.join(paths["data_test"], "manifest.csv"),
        preprocessor,
        is_training=False,
        use_face_detection=config["dataset"]["image"]["face_detection"]["enabled"]
    )
    
    # Handle class imbalance
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    mixed_precision: bool = False,
    gradient_clip: float = 1.0
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{total_loss/(batch_idx+1):.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False
):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if mixed_precision:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of fake class
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def compute_metrics(labels, preds, probs):
    """Compute comprehensive metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report
    )
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Image Deepfake Detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        config["image_detector"]["training"]["epochs"] = args.epochs
    
    # Setup
    device = torch.device(config["general"]["device"])
    logger.add(config["paths"]["logs_dir"] + "/train_image.log", rotation="10 MB")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config['image_detector']['training']}")
    
    # Set random seeds
    torch.manual_seed(config["general"]["random_seed"])
    import numpy as np
    np.random.seed(config["general"]["random_seed"])
    
    # Create preprocessor
    preprocessor = create_preprocessor_from_config(config)
    preprocessor.save(os.path.join(config["paths"]["models_dir"], "image_detector", "preprocessor.pkl"))
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        preprocessor,
        config["image_detector"]["training"]["batch_size"],
        config["general"]["num_workers"]
    )
    
    # Create model
    model = create_image_detector(config).to(device)
    logger.info(f"Model: {model.architecture}, params: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function
    training_cfg = config["image_detector"]["training"]
    if training_cfg["loss_function"] == "focal_loss":
        # Implement focal loss
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        criterion = FocalLoss()
    elif training_cfg["loss_function"] == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=training_cfg["loss_params"].get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if training_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"]
        )
    elif training_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            momentum=0.9,
            weight_decay=training_cfg["weight_decay"]
        )
    
    # Scheduler
    if training_cfg["scheduler"] == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg["scheduler_params"]["T_max"],
            eta_min=training_cfg["scheduler_params"]["eta_min"]
        )
    elif training_cfg["scheduler"] == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=training_cfg["scheduler_params"]["factor"],
            patience=training_cfg["scheduler_params"]["patience"],
            min_lr=training_cfg["scheduler_params"]["min_lr"]
        )
    else:
        scheduler = None
    
    # Mixed precision
    scaler = GradScaler() if training_cfg.get("mixed_precision", False) else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_auc = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint.get("metrics", {}).get("val_auc", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best AUC: {best_val_auc}")
    
    # Training loop
    epochs = training_cfg["epochs"]
    early_stopping_patience = training_cfg["early_stopping"]["patience"]
    early_stopping_counter = 0
    
    model_dir = Path(config["paths"]["models_dir"]) / "image_detector"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Set epoch for freeze/unfreeze
        model.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, training_cfg.get("mixed_precision", False),
            training_cfg.get("gradient_clip", 1.0)
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device,
            training_cfg.get("mixed_precision", False)
        )
        
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_auc = val_metrics["roc_auc"]
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
            
            save_image_detector(
                model,
                str(model_dir / "best_model.pth"),
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_auc": val_auc, **val_metrics},
                config=config
            )
            logger.info(f"Saved best model (AUC: {val_auc:.4f})")
        else:
            early_stopping_counter += 1
        
        # Save checkpoint
        save_image_detector(
            model,
            str(model_dir / f"checkpoint_epoch_{epoch}.pth"),
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_auc": val_auc, **val_metrics},
            config=config
        )
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
            break
    
    # Final evaluation on test set
    logger.info("\n=== Final Test Evaluation ===")
    best_model = load_image_detector(str(model_dir / "best_model.pth"), config, device)
    best_model.to(device)
    
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        best_model, test_loader, criterion, device
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logger.info(f"Test Metrics: {test_metrics}")
    
    # Save test metrics
    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()