"""
Video Deepfake Detector Models

Implements frame-level feature extraction with temporal aggregation
(LSTM, Transformer, Attention) for video-level deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TemporalAggregator(nn.Module):
    """Base class for temporal aggregation modules."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - batch of frame sequences
        Returns:
            [B, D_out] - aggregated representation
        """
        raise NotImplementedError


class MeanAggregator(TemporalAggregator):
    """Simple mean pooling over time."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class MaxAggregator(TemporalAggregator):
    """Max pooling over time."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values


class AttentionAggregator(TemporalAggregator):
    """Learnable attention-based temporal aggregation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__(input_dim, hidden_dim, dropout)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Weighted sum
        output = (x * attn_weights).sum(dim=1)  # [B, D]
        return output


class LSTMAggregator(TemporalAggregator):
    """LSTM-based temporal aggregation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__(input_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(lstm_output_dim, input_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        output = self.output_proj(self.dropout_layer(h_last))
        return output


class TransformerAggregator(TemporalAggregator):
    """Transformer-based temporal aggregation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_seq_len: int = 64
    ):
        super().__init__(input_dim, hidden_dim, dropout)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, input_dim)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token output
        cls_output = x[:, 0]  # [B, D]
        return self.dropout_layer(cls_output)


def create_temporal_aggregator(
    aggregator_type: str,
    input_dim: int,
    hidden_dim: int,
    **kwargs
) -> TemporalAggregator:
    """Factory function for temporal aggregators."""
    
    aggregators = {
        "mean": MeanAggregator,
        "max": MaxAggregator,
        "attention": AttentionAggregator,
        "lstm": LSTMAggregator,
        "transformer": TransformerAggregator,
    }
    
    if aggregator_type not in aggregators:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")
    
    return aggregators[aggregator_type](input_dim, hidden_dim, **kwargs)


class DeepfakeVideoDetector(nn.Module):
    """
    Video Deepfake Detector with frame-level backbone + temporal aggregation.
    
    Pipeline:
    1. Frame-level backbone (EfficientNet, ResNet, etc.) - shared weights
    2. Temporal aggregation (Mean, LSTM, Transformer, Attention)
    3. Classification head
    """
    
    def __init__(
        self,
        frame_backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone_epochs: int = 5,
        embedding_dim: int = 512,
        temporal_type: str = "lstm",
        temporal_hidden_dim: int = 256,
        temporal_layers: int = 2,
        temporal_dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        self.frame_backbone_name = frame_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Frame-level backbone (shared across frames)
        self.frame_backbone, frame_feature_dim = self._build_frame_backbone(
            frame_backbone, pretrained
        )
        
        # Project frame features to embedding_dim if needed
        if frame_feature_dim != embedding_dim:
            self.frame_projection = nn.Linear(frame_feature_dim, embedding_dim)
        else:
            self.frame_projection = nn.Identity()
        
        # Temporal aggregator
        self.temporal_aggregator = create_temporal_aggregator(
            temporal_type,
            input_dim=embedding_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            bidirectional=bidirectional,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # For Grad-CAM per frame
        self._frame_gradients = None
        self._frame_activations = None
        self._register_frame_hooks()
    
    def _build_frame_backbone(self, architecture: str, pretrained: bool):
        """Build frame-level backbone (reuse image detector logic)."""
        import torchvision.models as models
        
        if architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
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
            
        elif architecture == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            self.target_layer_name = "features.7"
            
        else:
            raise ValueError(f"Unknown frame backbone: {architecture}")
        
        return backbone, feature_dim
    
    def _register_frame_hooks(self):
        """Register hooks on frame backbone for per-frame Grad-CAM."""
        def forward_hook(module, input, output):
            self._frame_activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self._frame_gradients = grad_output[0].detach()
        
        target_layer = self._get_target_layer()
        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> Optional[nn.Module]:
        """Get target layer by name."""
        parts = self.target_layer_name.split(".")
        module = self.frame_backbone
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
        """Freeze frame backbone parameters."""
        for param in self.frame_backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze frame backbone parameters."""
        for param in self.frame_backbone.parameters():
            param.requires_grad = True
    
    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features for each frame independently.
        
        Args:
            frames: [B, T, C, H, W]
            
        Returns:
            features: [B, T, embedding_dim]
        """
        B, T, C, H, W = frames.shape
        
        # Reshape to process all frames at once: [B*T, C, H, W]
        frames_flat = frames.view(B * T, C, H, W)
        
        # Extract features
        with torch.set_grad_enabled(self.training):
            features = self.frame_backbone(frames_flat)  # [B*T, feature_dim]
        
        # Project to embedding_dim
        features = self.frame_projection(features)  # [B*T, embedding_dim]
        
        # Reshape back: [B, T, embedding_dim]
        features = features.view(B, T, -1)
        
        return features
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            frames: [B, T, C, H, W] or [B, C, H, W] (single frame)
            
        Returns:
            logits: [B, num_classes]
        """
        # Handle single frame input
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)  # [B, 1, C, H, W]
        
        # Extract frame features
        frame_features = self.extract_frame_features(frames)  # [B, T, D]
        
        # Temporal aggregation
        video_features = self.temporal_aggregator(frame_features)  # [B, D]
        
        # Classification
        logits = self.classifier(video_features)
        
        return logits
    
    def get_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Get per-frame features [B, T, D]."""
        return self.extract_frame_features(frames)
    
    def get_video_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Get video-level features [B, D]."""
        frame_features = self.extract_frame_features(frames)
        return self.temporal_aggregator(frame_features)
    
    def get_per_frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
        """Get logits for each frame independently [B, T, num_classes]."""
        frame_features = self.extract_frame_features(frames)
        B, T, D = frame_features.shape
        
        # Apply classifier to each frame
        frame_features_flat = frame_features.view(B * T, D)
        frame_logits = self.classifier(frame_features_flat)
        frame_logits = frame_logits.view(B, T, -1)
        
        return frame_logits
    
    def get_gradcam_activations(self) -> Optional[torch.Tensor]:
        return self._frame_activations
    
    def get_gradcam_gradients(self) -> Optional[torch.Tensor]:
        return self._frame_gradients
    
    def clear_gradcam(self):
        self._frame_activations = None
        self._frame_gradients = None


def create_video_detector(config: Dict) -> DeepfakeVideoDetector:
    """Create video detector from config dictionary."""
    model_cfg = config.get("video_detector", {})
    frame_cfg = model_cfg.get("frame_backbone", {})
    temporal_cfg = model_cfg.get("temporal", {})
    
    return DeepfakeVideoDetector(
        frame_backbone=frame_cfg.get("architecture", "efficientnet_b0"),
        pretrained=frame_cfg.get("pretrained", True),
        freeze_backbone_epochs=frame_cfg.get("freeze_backbone_epochs", 5),
        embedding_dim=frame_cfg.get("embedding_dim", 512),
        temporal_type=temporal_cfg.get("type", "lstm"),
        temporal_hidden_dim=temporal_cfg.get("hidden_dim", 256),
        temporal_layers=temporal_cfg.get("num_layers", 2),
        temporal_dropout=temporal_cfg.get("dropout", 0.3),
        bidirectional=temporal_cfg.get("bidirectional", True),
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("dropout", 0.4),
    )


def load_video_detector(
    model_path: str,
    config: Dict,
    device: torch.device = None
) -> DeepfakeVideoDetector:
    """Load trained video detector from checkpoint."""
    device = device or torch.device("cpu")
    
    model = create_video_detector(config)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def save_video_detector(
    model: DeepfakeVideoDetector,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    metrics: Dict = None,
    config: Dict = None,
):
    """Save video detector checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "frame_backbone": model.frame_backbone_name,
        "embedding_dim": model.embedding_dim,
        "num_classes": model.num_classes,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, path)