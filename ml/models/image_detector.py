"""
Image Deepfake Detector Models

Implements EfficientNet-B0, ResNet50, ConvNeXt-Tiny, and Xception backbones
with custom classification heads for binary deepfake detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DeepfakeImageDetector(nn.Module):
    """
    Image Deepfake Detector with configurable backbone.
    
    Supports:
    - EfficientNet-B0 (default, good balance of speed/accuracy)
    - ResNet50 (classic, well-understood)
    - ConvNeXt-Tiny (modern, strong performance)
    - Xception (good for deepfake detection specifically)
    """
    
    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone_epochs: int = 5,
        num_classes: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        self.architecture = architecture
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.current_epoch = 0
        
        # Build backbone and get feature dimension
        self.backbone, self.feature_dim = self._build_backbone(architecture, pretrained)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )
        
        # For Grad-CAM: register hook on last conv layer
        self._gradients = None
        self._activations = None
        self._register_hooks()
    
    def _build_backbone(self, architecture: str, pretrained: bool) -> tuple:
        """Build backbone and return (model, feature_dim)."""
        
        if architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            # Remove classifier
            backbone.classifier = nn.Identity()
            # Target layer for Grad-CAM
            self.target_layer_name = "features.7"
            
        elif architecture == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.target_layer_name = "features.7"
            
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.target_layer_name = "layer4"
            
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.target_layer_name = "layer4"
            
        elif architecture == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            self.target_layer_name = "features.7"
            
        elif architecture == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_small(weights=weights)
            feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            self.target_layer_name = "features.7"
            
        elif architecture == "xception":
            # Xception not in torchvision, use timm if available
            try:
                import timm
                backbone = timm.create_model("xception", pretrained=pretrained, num_classes=0)
                feature_dim = backbone.num_features
                self.target_layer_name = "block12"
            except ImportError:
                logger.warning("timm not installed, falling back to EfficientNet-B0")
                return self._build_backbone("efficientnet_b0", pretrained)
                
        elif architecture == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            feature_dim = backbone.hidden_dim
            # Remove classifier head
            backbone.heads = nn.Identity()
            self.target_layer_name = "encoder.layers.encoder_layer_11"
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return backbone, feature_dim
    
    def _register_hooks(self):
        """Register hooks for Grad-CAM."""
        def forward_hook(module, input, output):
            self._activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()
        
        # Find target layer
        target_layer = self._get_target_layer()
        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> Optional[nn.Module]:
        """Get target layer by name for Grad-CAM."""
        parts = self.target_layer_name.split(".")
        module = self.backbone
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                break
        return module
    
    def set_epoch(self, epoch: int):
        """Set current epoch for freeze/unfreeze logic."""
        self.current_epoch = epoch
        if epoch < self.freeze_backbone_epochs:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            logits: [B, num_classes]
        """
        features = self.backbone(x)  # [B, feature_dim]
        logits = self.classifier(features)
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings (before classifier)."""
        return self.backbone(x)
    
    def get_gradcam_activations(self) -> Optional[torch.Tensor]:
        """Get activations for Grad-CAM."""
        return self._activations
    
    def get_gradcam_gradients(self) -> Optional[torch.Tensor]:
        """Get gradients for Grad-CAM."""
        return self._gradients
    
    def clear_gradcam(self):
        """Clear stored Grad-CAM activations/gradients."""
        self._activations = None
        self._gradients = None


class EnsembleImageDetector(nn.Module):
    """Ensemble of multiple image detectors for improved robustness."""
    
    def __init__(
        self,
        architectures: List[str] = None,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        if architectures is None:
            architectures = ["efficientnet_b0", "resnet50", "convnext_tiny"]
        
        self.detectors = nn.ModuleList([
            DeepfakeImageDetector(
                architecture=arch,
                pretrained=pretrained,
                num_classes=num_classes,
                dropout=dropout,
                embedding_dim=embedding_dim,
            )
            for arch in architectures
        ])
        
        self.num_models = len(architectures)
        self.architectures = architectures
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models."""
        logits_list = [detector(x) for detector in self.detectors]
        # Average logits
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return avg_logits
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each model separately."""
        return [detector(x) for detector in self.detectors]


def create_image_detector(config: Dict) -> DeepfakeImageDetector:
    """Create image detector from config dictionary."""
    model_cfg = config.get("image_detector", {})
    
    return DeepfakeImageDetector(
        architecture=model_cfg.get("architecture", "efficientnet_b0"),
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone_epochs=model_cfg.get("freeze_backbone_epochs", 5),
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("dropout", 0.3),
        embedding_dim=model_cfg.get("embedding_dim", 512),
    )


def load_image_detector(
    model_path: str,
    config: Dict,
    device: torch.device = None
) -> DeepfakeImageDetector:
    """Load trained image detector from checkpoint."""
    device = device or torch.device("cpu")
    
    model = create_image_detector(config)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def save_image_detector(
    model: DeepfakeImageDetector,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    metrics: Dict = None,
    config: Dict = None,
):
    """Save image detector checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": model.architecture,
        "num_classes": model.num_classes,
        "embedding_dim": model.embedding_dim,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, path)


# Model metadata for versioning
def get_model_metadata(model: DeepfakeImageDetector, config: Dict) -> Dict[str, Any]:
    """Generate model metadata for versioning."""
    return {
        "model_name": "DeepfakeImageDetector",
        "version": "1.0.0",
        "architecture": model.architecture,
        "num_classes": model.num_classes,
        "embedding_dim": model.embedding_dim,
        "freeze_backbone_epochs": model.freeze_backbone_epochs,
        "training_config": config.get("image_detector", {}).get("training", {}),
        "dataset_config": config.get("dataset", {}).get("image", {}),
        "class_mapping": {0: "Real", 1: "Fake"},
        "preprocessing_version": "1.0",
    }