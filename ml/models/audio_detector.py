"""
Audio Deepfake Detector Models

Implements RawNet2, Wav2Vec2, Conformer, and AST backbones
for synthetic/manipulated speech detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import logging
import math

logger = logging.getLogger(__name__)


class SincConv(nn.Module):
    """Sinc-based convolution for RawNet2."""
    
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        in_channels: int = 1,
        sample_rate: int = 16000,
        min_freq: float = 50.0,
        max_freq: float = 8000.0
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        
        # Initialize sinc filters
        # Frequencies in Hz
        low_hz = torch.linspace(min_freq, max_freq, out_channels // 2)
        high_hz = low_hz + (max_freq - min_freq) / (out_channels // 2) * torch.rand(out_channels // 2)
        
        # Convert to mel scale for better initialization
        low_mel = 2595 * torch.log10(1 + low_hz / 700)
        high_mel = 2595 * torch.log10(1 + high_hz / 700)
        
        self.low_hz = nn.Parameter(low_hz)
        self.high_hz = nn.Parameter(high_hz)
        
        # Hamming window
        n = torch.arange(kernel_size) - (kernel_size - 1) / 2
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1)))
        
        # Learnable amplitude
        self.amplitude = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] - raw waveform
        Returns:
            [B, out_channels, T']
        """
        # Compute sinc filters on the fly
        device = x.device
        n = torch.arange(self.kernel_size, device=device) - (self.kernel_size - 1) / 2
        
        # Low and high frequency filters
        low = 2 * math.pi * self.low_hz / self.sample_rate
        high = 2 * math.pi * self.high_hz / self.sample_rate
        
        # Bandpass filters
        band_filters = []
        for i in range(self.out_channels // 2):
            low_i = low[i]
            high_i = high[i]
            
            # Sinc function for bandpass
            filter_i = (torch.sinc(high_i * n / math.pi) - torch.sinc(low_i * n / math.pi)) * self.window
            band_filters.append(filter_i)
        
        # Stack filters
        filters = torch.stack(band_filters, dim=0)  # [out_channels//2, kernel_size]
        filters = filters * self.amplitude[:self.out_channels//2].view(-1, 1)
        
        # Apply convolution
        # x: [B, 1, T] -> [B, out_channels//2, T']
        out = F.conv1d(x, filters.unsqueeze(1), padding=self.kernel_size // 2)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block for RawNet2."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.3)
        return out


class RawNet2(nn.Module):
    """
    RawNet2: End-to-end speaker verification / anti-spoofing from raw waveforms.
    
    Paper: "RawNet2: Improved RawNet for Speaker Verification" (2020)
    Adapted for deepfake audio detection.
    """
    
    def __init__(
        self,
        sinc_conv: Dict = None,
        resblock: Dict = None,
        gru: Dict = None,
        fc: Dict = None,
        num_classes: int = 2,
        embedding_dim: int = 256,
    ):
        super().__init__()
        
        sinc_cfg = sinc_conv or {"out_channels": 70, "kernel_size": 1024, "in_channels": 1}
        resblock_cfg = resblock or {"channels": [70, 70, 140, 140, 280, 280]}
        gru_cfg = gru or {"hidden_dim": 1024, "num_layers": 3}
        fc_cfg = fc or {"hidden_dim": 1024}
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # SincConv layer
        self.sinc_conv = SincConv(
            out_channels=sinc_cfg["out_channels"],
            kernel_size=sinc_cfg["kernel_size"],
            in_channels=sinc_cfg["in_channels"]
        )
        self.bn_sinc = nn.BatchNorm1d(sinc_cfg["out_channels"])
        
        # Residual blocks
        channels = resblock_cfg["channels"]
        self.res_blocks = nn.ModuleList()
        in_ch = sinc_cfg["out_channels"]
        
        for i, out_ch in enumerate(channels):
            stride = 2 if i % 2 == 1 else 1  # Downsample every other block
            self.res_blocks.append(ResidualBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=channels[-1],
            hidden_size=gru_cfg["hidden_dim"],
            num_layers=gru_cfg["num_layers"],
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers
        gru_output_dim = gru_cfg["hidden_dim"] * 2  # bidirectional
        self.fc1 = nn.Linear(gru_output_dim, fc_cfg["hidden_dim"])
        self.bn_fc1 = nn.BatchNorm1d(fc_cfg["hidden_dim"])
        self.fc2 = nn.Linear(fc_cfg["hidden_dim"], embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] or [B, T] - raw waveform
            
        Returns:
            logits: [B, num_classes]
        """
        # Ensure [B, 1, T] shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # SincConv
        x = self.sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), kernel_size=3, stride=1, padding=1)
        x = self.bn_sinc(x)
        x = F.leaky_relu(x, 0.3)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Prepare for GRU: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # GRU
        x, _ = self.gru(x)
        
        # Use last hidden state
        x = x[:, -1, :]  # [B, hidden_dim*2]
        
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.fc2(x)  # [B, embedding_dim]
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings before classification."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), kernel_size=3, stride=1, padding=1)
        x = self.bn_sinc(x)
        x = F.leaky_relu(x, 0.3)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.fc2(x)
        
        return x


class AMSoftmax(nn.Module):
    """Additive Margin Softmax for speaker verification / anti-spoofing."""
    
    def __init__(self, in_features: int, out_features: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] - embeddings
            labels: [B] - ground truth labels
        Returns:
            logits: [B, C] - scaled cosine similarity with margin
        """
        # Normalize embeddings and weights
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cos_theta = F.linear(x, w)  # [B, C]
        cos_theta = cos_theta.clamp(-1, 1)
        
        # Apply margin to target class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        cos_theta_m = cos_theta - one_hot * self.margin
        logits = cos_theta_m * self.scale
        
        return logits


class Wav2Vec2Detector(nn.Module):
    """Wav2Vec2-based detector for audio deepfake detection."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        embedding_dim: int = 256,
        freeze_feature_encoder: bool = True,
        freeze_transformer_layers: int = 0,
    ):
        super().__init__()
        
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Config
        except ImportError:
            raise ImportError("transformers library required for Wav2Vec2. Install with: pip install transformers")
        
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze feature encoder
        if freeze_feature_encoder:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Freeze transformer layers
        if freeze_transformer_layers > 0:
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < freeze_transformer_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes)
        )
        
        # Projector for embeddings
        self.embedding_proj = nn.Linear(hidden_size, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] - raw waveform (16kHz)
        """
        outputs = self.wav2vec2(x)
        last_hidden = outputs.last_hidden_state  # [B, T', D]
        
        # Mean pooling over time
        pooled = last_hidden.mean(dim=1)  # [B, D]
        
        logits = self.classifier(pooled)
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2(x)
        last_hidden = outputs.last_hidden_state
        pooled = last_hidden.mean(dim=1)
        return self.embedding_proj(pooled)


class ConformerDetector(nn.Module):
    """Conformer-based audio detector."""
    
    def __init__(
        self,
        input_dim: int = 80,  # mel-spectrogram features
        encoder_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        num_classes: int = 2,
        embedding_dim: int = 256,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        
        # Conformer encoder layers
        self.encoder_layers = nn.ModuleList([
            ConformerBlock(encoder_dim, num_heads, ff_dim, dropout, conv_kernel_size)
            for _ in range(num_layers)
        ])
        
        self.embedding_proj = nn.Linear(encoder_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - mel-spectrogram or [B, D, T] - conv features
        """
        # Ensure [B, T, D]
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            # Assume [B, D, T] -> transpose
            if x.shape[1] < x.shape[2]:
                x = x.transpose(1, 2)
        
        x = self.input_proj(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Mean pooling
        x = x.mean(dim=1)  # [B, D]
        
        embeddings = self.embedding_proj(x)
        logits = self.classifier(embeddings)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            if x.shape[1] < x.shape[2]:
                x = x.transpose(1, 2)
        
        x = self.input_proj(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.embedding_proj(x)


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(
        self,
        encoder_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        conv_kernel_size: int,
    ):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = FeedForwardModule(encoder_dim, ff_dim, dropout)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            encoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(encoder_dim)
        self.dropout_attn = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = ConvolutionModule(encoder_dim, conv_kernel_size, dropout)
        
        # Feed-forward module 2
        self.ff2 = FeedForwardModule(encoder_dim, ff_dim, dropout)
        
        self.norm_final = nn.LayerNorm(encoder_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FF1
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.norm_attn(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout_attn(x)
        x = residual + x
        
        # Conv
        x = x + self.conv_module(x)
        
        # FF2
        x = x + 0.5 * self.ff2(x)
        
        # Final norm
        x = self.norm_final(x)
        
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer."""
    
    def __init__(self, encoder_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, encoder_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(self, encoder_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Conv1d(encoder_dim, encoder_dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(
                encoder_dim, encoder_dim,
                kernel_size, padding=(kernel_size - 1) // 2,
                groups=encoder_dim
            ),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Conv1d(encoder_dim, encoder_dim, 1),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T] for conv1d
        x = x.transpose(1, 2)
        x = self.seq(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class ASTDetector(nn.Module):
    """Audio Spectrogram Transformer (AST) detector."""
    
    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_classes: int = 2,
        embedding_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        try:
            from transformers import ASTModel, ASTConfig
        except ImportError:
            raise ImportError("transformers library required for AST. Install with: pip install transformers")
        
        self.config = ASTConfig.from_pretrained(model_name)
        self.ast = ASTModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.ast.parameters():
                param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        
        # Classification head (uses CLS token)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes)
        )
        
        self.embedding_proj = nn.Linear(hidden_size, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] - mel-spectrogram (T=time frames, F=mel bins)
        """
        outputs = self.ast(input_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # [B, D]
        
        logits = self.classifier(cls_token)
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.ast(input_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.embedding_proj(cls_token)


def create_audio_detector(config: Dict) -> nn.Module:
    """Create audio detector from config dictionary."""
    model_cfg = config.get("audio_detector", {})
    architecture = model_cfg.get("architecture", "rawnet2")
    
    if architecture == "rawnet2":
        rawnet_cfg = model_cfg.get("rawnet2", {})
        return RawNet2(
            sinc_conv=rawnet_cfg.get("sinc_conv"),
            resblock=rawnet_cfg.get("resblock"),
            gru=rawnet_cfg.get("gru"),
            fc=rawnet_cfg.get("fc"),
            num_classes=model_cfg.get("num_classes", 2),
            embedding_dim=model_cfg.get("embedding_dim", 256),
        )
    elif architecture == "wav2vec2":
        return Wav2Vec2Detector(
            model_name=model_cfg.get("wav2vec2_model", "facebook/wav2vec2-base"),
            num_classes=model_cfg.get("num_classes", 2),
            embedding_dim=model_cfg.get("embedding_dim", 256),
            freeze_feature_encoder=model_cfg.get("freeze_feature_encoder", True),
            freeze_transformer_layers=model_cfg.get("freeze_transformer_layers", 6),
        )
    elif architecture == "conformer":
        return ConformerDetector(
            input_dim=model_cfg.get("input_dim", 80),
            encoder_dim=model_cfg.get("encoder_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 4),
            num_classes=model_cfg.get("num_classes", 2),
            embedding_dim=model_cfg.get("embedding_dim", 256),
        )
    elif architecture == "ast":
        return ASTDetector(
            model_name=model_cfg.get("ast_model", "MIT/ast-finetuned-audioset-10-10-0.4593"),
            num_classes=model_cfg.get("num_classes", 2),
            embedding_dim=model_cfg.get("embedding_dim", 256),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )
    else:
        raise ValueError(f"Unknown audio architecture: {architecture}")


def load_audio_detector(
    model_path: str,
    config: Dict,
    device: torch.device = None
) -> nn.Module:
    """Load trained audio detector from checkpoint."""
    device = device or torch.device("cpu")
    
    model = create_audio_detector(config)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def save_audio_detector(
    model: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    metrics: Dict = None,
    config: Dict = None,
):
    """Save audio detector checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": config.get("audio_detector", {}).get("architecture", "rawnet2"),
        "num_classes": config.get("audio_detector", {}).get("num_classes", 2),
        "embedding_dim": config.get("audio_detector", {}).get("embedding_dim", 256),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, path)