"""
Multimodal Fusion Model for Deepfake Detection

Combines visual (image/video) and audio features for joint deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FusionModule(nn.Module):
    """Base class for multimodal fusion."""
    
    def __init__(self, visual_dim: int, audio_dim: int, output_dim: int):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [B, D_v] or [B, T, D_v]
            audio_features: [B, D_a] or [B, T, D_a]
        Returns:
            fused_features: [B, D_out]
        """
        raise NotImplementedError


class ConcatFusion(FusionModule):
    """Simple concatenation fusion."""
    
    def __init__(self, visual_dim: int, audio_dim: int, output_dim: int):
        super().__init__(visual_dim, audio_dim, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        # Handle sequence inputs by pooling
        if visual_features.dim() == 3:
            visual_features = visual_features.mean(dim=1)
        if audio_features.dim() == 3:
            audio_features = audio_features.mean(dim=1)
        
        fused = torch.cat([visual_features, audio_features], dim=1)
        return self.projection(fused)


class AttentionFusion(FusionModule):
    """Cross-modal attention fusion."""
    
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__(visual_dim, audio_dim, output_dim)
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-attention: visual attends to audio and vice versa
        self.visual_to_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(inplace=True),
        )
        
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        # Project to common dimension
        v = self.visual_proj(visual_features)  # [B, T_v, H] or [B, H]
        a = self.audio_proj(audio_features)    # [B, T_a, H] or [B, H]
        
        # Ensure sequence dimension
        if v.dim() == 2:
            v = v.unsqueeze(1)  # [B, 1, H]
        if a.dim() == 2:
            a = a.unsqueeze(1)  # [B, 1, H]
        
        # Cross-attention
        v_attended, _ = self.visual_to_audio(v, a, a)
        a_attended, _ = self.audio_to_visual(a, v, v)
        
        v_attended = self.norm_v(v_attended + v)
        a_attended = self.norm_a(a_attended + a)
        
        # Pool and concatenate
        v_pooled = v_attended.mean(dim=1)
        a_pooled = a_attended.mean(dim=1)
        
        fused = torch.cat([v_pooled, a_pooled], dim=1)
        return self.output_proj(fused)


class BilinearFusion(FusionModule):
    """Bilinear pooling fusion."""
    
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        output_dim: int,
        rank: int = 64
    ):
        super().__init__(visual_dim, audio_dim, output_dim)
        
        # Low-rank bilinear pooling
        self.visual_factor = nn.Linear(visual_dim, rank, bias=False)
        self.audio_factor = nn.Linear(audio_dim, rank, bias=False)
        self.output_proj = nn.Linear(rank, output_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        if visual_features.dim() == 3:
            visual_features = visual_features.mean(dim=1)
        if audio_features.dim() == 3:
            audio_features = audio_features.mean(dim=1)
        
        v = self.visual_factor(visual_features)  # [B, R]
        a = self.audio_factor(audio_features)    # [B, R]
        
        # Element-wise product (Hadamard)
        fused = v * a
        return self.output_proj(fused)


class GatedFusion(FusionModule):
    """Gated fusion with modality-specific gates."""
    
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__(visual_dim, audio_dim, output_dim)
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Gates
        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        if visual_features.dim() == 3:
            visual_features = visual_features.mean(dim=1)
        if audio_features.dim() == 3:
            audio_features = audio_features.mean(dim=1)
        
        v = self.visual_proj(visual_features)
        a = self.audio_proj(audio_features)
        
        # Apply gates
        v_gated = v * self.visual_gate(v)
        a_gated = a * self.audio_gate(a)
        
        fused = torch.cat([v_gated, a_gated], dim=1)
        return self.fusion(fused)


def create_fusion_module(
    fusion_type: str,
    visual_dim: int,
    audio_dim: int,
    output_dim: int,
    **kwargs
) -> FusionModule:
    """Factory function for fusion modules."""
    
    modules = {
        "concatenate": ConcatFusion,
        "attention": AttentionFusion,
        "bilinear": BilinearFusion,
        "gated": GatedFusion,
    }
    
    if fusion_type not in modules:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return modules[fusion_type](visual_dim, audio_dim, output_dim, **kwargs)


class MultimodalDeepfakeDetector(nn.Module):
    """
    Multimodal Deepfake Detector combining visual and audio streams.
    
    Architecture:
    1. Visual branch: Video detector (frame backbone + temporal agg) OR Image detector
    2. Audio branch: RawNet2 / Wav2Vec2 / Conformer / AST
    3. Fusion: Concatenation / Attention / Bilinear / Gated
    4. Joint classification head
    """
    
    def __init__(
        self,
        # Visual branch
        visual_branch: str = "video",  # "video" or "image"
        visual_backbone: str = "efficientnet_b0",
        visual_pretrained: bool = True,
        visual_embedding_dim: int = 512,
        temporal_type: str = "lstm",
        temporal_hidden_dim: int = 256,
        
        # Audio branch
        audio_architecture: str = "rawnet2",
        audio_embedding_dim: int = 256,
        audio_config: Dict = None,
        
        # Fusion
        fusion_type: str = "concatenate",
        fusion_hidden_dims: List[int] = None,
        fusion_dropout: float = 0.4,
        fusion_activation: str = "relu",
        
        # Classifier
        num_classes: int = 2,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        
        self.visual_branch = visual_branch
        self.audio_architecture = audio_architecture
        self.num_classes = num_classes
        
        # Visual branch
        if visual_branch == "video":
            from ml.models.video_detector import DeepfakeVideoDetector
            self.visual_detector = DeepfakeVideoDetector(
                frame_backbone=visual_backbone,
                pretrained=visual_pretrained,
                embedding_dim=visual_embedding_dim,
                temporal_type=temporal_type,
                temporal_hidden_dim=temporal_hidden_dim,
                num_classes=num_classes,  # Not used directly
                dropout=0.0,  # No classifier in backbone
            )
            # Remove classifier from visual detector
            self.visual_detector.classifier = nn.Identity()
            visual_feat_dim = visual_embedding_dim
            
        elif visual_branch == "image":
            from ml.models.image_detector import DeepfakeImageDetector
            self.visual_detector = DeepfakeImageDetector(
                architecture=visual_backbone,
                pretrained=visual_pretrained,
                embedding_dim=visual_embedding_dim,
                num_classes=num_classes,
                dropout=0.0,
            )
            self.visual_detector.classifier = nn.Identity()
            visual_feat_dim = visual_embedding_dim
            
        else:
            raise ValueError(f"Unknown visual branch: {visual_branch}")
        
        # Audio branch
        self.audio_detector = self._build_audio_detector(
            audio_architecture, audio_embedding_dim, audio_config or {}
        )
        audio_feat_dim = audio_embedding_dim
        
        # Fusion module
        self.fusion = create_fusion_module(
            fusion_type,
            visual_feat_dim,
            audio_feat_dim,
            fusion_hidden_dims[0] if fusion_hidden_dims else 512,
            **kwargs
        )
        
        # Fusion hidden layers
        fusion_layers = []
        dims = fusion_hidden_dims or [512, 256]
        in_dim = fusion_hidden_dims[0] if fusion_hidden_dims else 512
        
        for out_dim in dims[1:]:
            if fusion_activation == "relu":
                act = nn.ReLU(inplace=True)
            elif fusion_activation == "gelu":
                act = nn.GELU()
            elif fusion_activation == "silu":
                act = nn.SiLU()
            else:
                act = nn.ReLU(inplace=True)
            
            fusion_layers.extend([
                nn.Linear(in_dim, out_dim),
                act,
                nn.Dropout(fusion_dropout),
            ])
            in_dim = out_dim
        
        self.fusion_layers = nn.Sequential(*fusion_layers) if fusion_layers else nn.Identity()
        final_dim = dims[-1] if dims else in_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(final_dim, num_classes)
        )
    
    def _build_audio_detector(
        self,
        architecture: str,
        embedding_dim: int,
        config: Dict
    ) -> nn.Module:
        """Build audio detector based on architecture."""
        
        if architecture == "rawnet2":
            rawnet_cfg = config.get("rawnet2", {})
            from ml.models.audio_detector import RawNet2
            return RawNet2(
                sinc_conv=rawnet_cfg.get("sinc_conv"),
                resblock=rawnet_cfg.get("resblock"),
                gru=rawnet_cfg.get("gru"),
                fc=rawnet_cfg.get("fc"),
                num_classes=2,
                embedding_dim=embedding_dim,
            )
        elif architecture == "wav2vec2":
            from ml.models.audio_detector import Wav2Vec2Detector
            return Wav2Vec2Detector(
                model_name=config.get("wav2vec2_model", "facebook/wav2vec2-base"),
                num_classes=2,
                embedding_dim=embedding_dim,
            )
        elif architecture == "conformer":
            from ml.models.audio_detector import ConformerDetector
            return ConformerDetector(
                input_dim=config.get("input_dim", 80),
                encoder_dim=config.get("encoder_dim", 256),
                num_classes=2,
                embedding_dim=embedding_dim,
            )
        elif architecture == "ast":
            from ml.models.audio_detector import ASTDetector
            return ASTDetector(
                model_name=config.get("ast_model", "MIT/ast-finetuned-audioset-10-10-0.4593"),
                num_classes=2,
                embedding_dim=embedding_dim,
            )
        else:
            raise ValueError(f"Unknown audio architecture: {architecture}")
    
    def forward(
        self,
        visual_input: torch.Tensor,
        audio_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visual_input: Video frames [B, T, C, H, W] or Image [B, C, H, W]
            audio_input: Waveform [B, 1, T] or [B, T] or Mel-spec [B, T, F]
            
        Returns:
            logits: [B, num_classes]
        """
        # Extract visual features
        if self.visual_branch == "video":
            visual_feats = self.visual_detector.get_video_features(visual_input)
        else:
            visual_feats = self.visual_detector.get_embeddings(visual_input)
        
        # Extract audio features
        if hasattr(self.audio_detector, 'get_embeddings'):
            audio_feats = self.audio_detector.get_embeddings(audio_input)
        else:
            # For RawNet2, use forward without classifier
            audio_feats = self._get_audio_embeddings(audio_input)
        
        # Fusion
        fused = self.fusion(visual_feats, audio_feats)
        fused = self.fusion_layers(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def _get_audio_embeddings(self, audio_input: torch.Tensor) -> torch.Tensor:
        """Get embeddings from RawNet2-style models."""
        # This handles RawNet2 which doesn't have get_embeddings method exposed
        if isinstance(self.audio_detector, RawNet2):
            return self.audio_detector.get_embeddings(audio_input)
        else:
            return self.audio_detector(audio_input)
    
    def get_visual_features(self, visual_input: torch.Tensor) -> torch.Tensor:
        """Get visual features only."""
        if self.visual_branch == "video":
            return self.visual_detector.get_video_features(visual_input)
        else:
            return self.visual_detector.get_embeddings(visual_input)
    
    def get_audio_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        """Get audio features only."""
        if hasattr(self.audio_detector, 'get_embeddings'):
            return self.audio_detector.get_embeddings(audio_input)
        else:
            return self._get_audio_embeddings(audio_input)
    
    def get_fused_features(
        self,
        visual_input: torch.Tensor,
        audio_input: torch.Tensor
    ) -> torch.Tensor:
        """Get fused features before classification."""
        visual_feats = self.get_visual_features(visual_input)
        audio_feats = self.get_audio_features(audio_input)
        fused = self.fusion(visual_feats, audio_feats)
        return self.fusion_layers(fused)


def create_multimodal_detector(config: Dict) -> MultimodalDeepfakeDetector:
    """Create multimodal detector from config dictionary."""
    model_cfg = config.get("multimodal_detector", {})
    visual_cfg = model_cfg.get("visual_branch", {})
    audio_cfg = model_cfg.get("audio_branch", {})
    fusion_cfg = model_cfg.get("fusion", {})
    classifier_cfg = model_cfg.get("classifier", {})
    
    return MultimodalDeepfakeDetector(
        visual_branch=visual_cfg.get("backbone", "video"),
        visual_backbone=visual_cfg.get("backbone", "efficientnet_b0"),
        visual_pretrained=visual_cfg.get("pretrained", True),
        visual_embedding_dim=visual_cfg.get("embedding_dim", 512),
        temporal_type=config.get("video_detector", {}).get("temporal", {}).get("type", "lstm"),
        temporal_hidden_dim=config.get("video_detector", {}).get("temporal", {}).get("hidden_dim", 256),
        
        audio_architecture=audio_cfg.get("architecture", "rawnet2"),
        audio_embedding_dim=audio_cfg.get("embedding_dim", 256),
        audio_config=config.get("audio_detector", {}),
        
        fusion_type=fusion_cfg.get("fusion_type", "concatenate"),
        fusion_hidden_dims=fusion_cfg.get("hidden_dims", [512, 256]),
        fusion_dropout=fusion_cfg.get("dropout", 0.4),
        fusion_activation=fusion_cfg.get("activation", "relu"),
        
        num_classes=classifier_cfg.get("num_classes", 2),
        classifier_dropout=classifier_cfg.get("dropout", 0.3),
    )


def load_multimodal_detector(
    model_path: str,
    config: Dict,
    device: torch.device = None
) -> MultimodalDeepfakeDetector:
    """Load trained multimodal detector from checkpoint."""
    device = device or torch.device("cpu")
    
    model = create_multimodal_detector(config)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def save_multimodal_detector(
    model: MultimodalDeepfakeDetector,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    metrics: Dict = None,
    config: Dict = None,
):
    """Save multimodal detector checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "visual_branch": model.visual_branch,
        "audio_architecture": model.audio_architecture,
        "num_classes": model.num_classes,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, path)