"""
Evaluation Module for Deepfake Detection Models

Comprehensive evaluation including metrics, robustness testing, threshold analysis.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    det_curve
)
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str] = None
    ):
        self.model = model
        self.device = device
        self.class_names = class_names or ["Real", "Fake"]
        self.model.eval()
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        criterion: nn.Module = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Handle different batch formats
                if len(batch) == 2:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                elif len(batch) == 3:
                    inputs, _, labels = batch  # (video, audio, label) or (waveform, mel, label)
                    inputs = inputs[0].to(self.device) if isinstance(inputs, tuple) else inputs.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                
                labels = labels.to(self.device)
                
                # Forward pass
                if isinstance(inputs, tuple):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                # Loss
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                # Predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of fake
                all_logits.extend(outputs.cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics["avg_loss"] = total_loss / len(dataloader) if criterion else None
        
        if return_details:
            return {
                "metrics": metrics,
                "predictions": all_preds,
                "labels": all_labels,
                "probabilities": all_probs,
                "logits": all_logits
            }
        
        return metrics
    
    def _compute_metrics(
        self,
        labels: List[int],
        preds: List[int],
        probs: List[float]
    ) -> Dict[str, Any]:
        """Compute all metrics."""
        from sklearn.metrics import roc_curve
        
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="binary", zero_division=0),
            "recall": recall_score(labels, preds, average="binary", zero_division=0),
            "f1": f1_score(labels, preds, average="binary", zero_division=0),
            "roc_auc": roc_auc_score(labels, probs),
            "pr_auc": average_precision_score(labels, probs),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            "classification_report": classification_report(labels, preds, target_names=self.class_names, output_dict=True),
        }
        
        # EER (Equal Error Rate)
        fpr, tpr, thresholds = roc_curve(labels, probs)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        metrics["eer"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        metrics["eer_threshold"] = float(thresholds[eer_idx])
        
        return metrics
    
    def threshold_analysis(
        self,
        dataloader: DataLoader,
        num_thresholds: int = 100
    ) -> Dict[str, Any]:
        """Analyze performance across different thresholds."""
        self.model.eval()
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Threshold Analysis"):
                if len(batch) == 2:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                elif len(batch) == 3:
                    inputs, _, labels = batch
                    inputs = inputs[0].to(self.device) if isinstance(inputs, tuple) else inputs.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format")
                
                labels = labels.to(self.device)
                
                if isinstance(inputs, tuple):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        labels = np.array(all_labels)
        probs = np.array(all_probs)
        
        thresholds = np.linspace(0, 1, num_thresholds)
        results = []
        
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            tn = np.sum((preds == 0) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results.append({
                "threshold": float(thresh),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            })
        
        # Find optimal thresholds
        f1_scores = [r["f1"] for r in results]
        best_f1_idx = np.argmax(f1_scores)
        
        # Youden's J statistic (sensitivity + specificity - 1)
        youden_j = [r["recall"] + (1 - r["fpr"]) - 1 for r in results]
        best_youden_idx = np.argmax(youden_j)
        
        return {
            "thresholds": results,
            "best_f1_threshold": results[best_f1_idx],
            "best_youden_threshold": results[best_youden_idx],
        }
    
    def robustness_test(
        self,
        dataloader: DataLoader,
        transformations: List[Dict[str, Any]],
        criterion: nn.Module = None
    ) -> Dict[str, Any]:
        """Test model robustness under various transformations."""
        from ml.preprocessing.image_preprocessing import apply_jpeg_compression
        from PIL import Image
        import torch
        import torchvision.transforms as T
        
        self.model.eval()
        results = {}
        
        for transform_config in transformations:
            transform_name = transform_config["name"]
            params = transform_config.get("params", [])
            
            logger.info(f"Testing robustness: {transform_name}")
            transform_results = []
            
            for param in params:
                # Create modified dataloader with transformation
                # This is simplified - in practice you'd apply transform on-the-fly
                pass
            
            results[transform_name] = transform_results
        
        return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str = None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(labels: np.ndarray, probs: np.ndarray, save_path: str = None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_det_curve(labels: np.ndarray, probs: np.ndarray, save_path: str = None):
    """Plot Detection Error Tradeoff (DET) curve."""
    fpr, fnr, _ = det_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('DET Curve')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(threshold_results: Dict, save_path: str = None):
    """Plot metrics vs threshold."""
    results = threshold_results["thresholds"]
    thresholds = [r["threshold"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy, label='Accuracy')
    plt.plot(thresholds, precision, label='Precision')
    plt.plot(thresholds, recall, label='Recall')
    plt.plot(thresholds, f1, label='F1')
    plt.axvline(threshold_results["best_f1_threshold"]["threshold"], color='red', linestyle='--', label=f'Best F1')
    plt.axvline(threshold_results["best_youden_threshold"]["threshold"], color='green', linestyle='--', label=f'Best Youden')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_models(model_results: Dict[str, Dict], save_path: str = None):
    """Compare multiple models."""
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "eer"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_names):
        ax = axes[i]
        model_names = list(model_results.keys())
        values = [model_results[name]["metrics"].get(metric, 0) for name in model_names]
        
        bars = ax.bar(model_names, values)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # Remove empty subplot
    if len(metrics_names) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(
    model_name: str,
    metrics: Dict,
    threshold_analysis: Dict = None,
    robustness_results: Dict = None,
    save_path: str = None
) -> str:
    """Generate comprehensive evaluation report."""
    
    report = f"""
# Evaluation Report: {model_name}

## Overall Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1 Score**: {metrics.get('f1', 0):.4f}
- **ROC-AUC**: {metrics.get('roc_auc', 0):.4f}
- **PR-AUC**: {metrics.get('pr_auc', 0):.4f}
- **EER**: {metrics.get('eer', 0):.4f}
- **EER Threshold**: {metrics.get('eer_threshold', 0):.4f}

## Confusion Matrix
```
{np.array(metrics.get('confusion_matrix', [[0,0],[0,0]]))}
```

## Per-Class Metrics
"""
    
    if 'classification_report' in metrics:
        cr = metrics['classification_report']
        for cls in ['Real', 'Fake']:
            if cls in cr:
                report += f"""
### {cls}
- Precision: {cr[cls]['precision']:.4f}
- Recall: {cr[cls]['recall']:.4f}
- F1: {cr[cls]['f1-score']:.4f}
- Support: {cr[cls]['support']}
"""
    
    if threshold_analysis:
        report += f"""
## Threshold Analysis
- **Best F1 Threshold**: {threshold_analysis['best_f1_threshold']['threshold']:.4f}
  - F1: {threshold_analysis['best_f1_threshold']['f1']:.4f}
  - Precision: {threshold_analysis['best_f1_threshold']['precision']:.4f}
  - Recall: {threshold_analysis['best_f1_threshold']['recall']:.4f}

- **Best Youden Threshold**: {threshold_analysis['best_youden_threshold']['threshold']:.4f}
  - Sensitivity: {threshold_analysis['best_youden_threshold']['recall']:.4f}
  - Specificity: {1 - threshold_analysis['best_youden_threshold']['fpr']:.4f}
"""
    
    if robustness_results:
        report += f"""
## Robustness Testing
"""
        for transform, results in robustness_results.items():
            report += f"\n### {transform}\n"
            for r in results:
                report += f"- {r}\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def evaluate_all_models(
    config: Dict,
    model_paths: Dict[str, str],
    dataloaders: Dict[str, DataLoader],
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate all trained models."""
    from ml.models.image_detector import load_image_detector
    from ml.models.video_detector import load_video_detector
    from ml.models.audio_detector import load_audio_detector
    from ml.models.multimodal_detector import load_multimodal_detector
    
    loaders = {
        "image": load_image_detector,
        "video": load_video_detector,
        "audio": load_audio_detector,
        "multimodal": load_multimodal_detector,
    }
    
    results = {}
    
    for model_name, model_path in model_paths.items():
        if model_name not in loaders:
            logger.warning(f"Unknown model type: {model_name}")
            continue
        
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        logger.info(f"Evaluating {model_name}...")
        
        model = loaders[model_name](model_path, config, device)
        evaluator = ModelEvaluator(model, device)
        
        # Use appropriate dataloader
        loader = dataloaders.get(model_name)
        if loader is None:
            logger.warning(f"No dataloader for {model_name}")
            continue
        
        metrics = evaluator.evaluate_dataset(loader)
        threshold_analysis = evaluator.threshold_analysis(loader)
        
        results[model_name] = {
            "metrics": metrics,
            "threshold_analysis": threshold_analysis
        }
        
        logger.info(f"{model_name} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    
    return results