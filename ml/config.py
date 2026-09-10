"""
Configuration management for ML pipeline
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class ImageConfig:
    target_size: tuple = (224, 224)
    supported_formats: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    max_file_size_mb: int = 10
    normalization_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalization_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    face_detection_enabled: bool = True
    min_face_size: int = 64
    face_confidence_threshold: float = 0.9


@dataclass
class VideoConfig:
    supported_formats: List[str] = field(default_factory=lambda: [".mp4", ".mov", ".avi"])
    max_file_size_mb: int = 100
    max_duration_seconds: int = 60
    frame_sampling_method: str = "uniform"
    num_frames: int = 16
    min_frames: int = 8
    face_detection_enabled: bool = True
    min_face_size: int = 64
    face_confidence_threshold: float = 0.9


@dataclass
class AudioConfig:
    supported_formats: List[str] = field(default_factory=lambda: [".wav", ".mp3", ".m4a"])
    max_file_size_mb: int = 20
    max_duration_seconds: int = 30
    sample_rate: int = 16000
    segment_length: int = 4
    spectrogram_n_fft: int = 512
    spectrogram_hop_length: int = 160
    spectrogram_n_mels: int = 80
    spectrogram_target_length: int = 400


@dataclass
class DatasetConfig:
    sources: List[Dict[str, Any]] = field(default_factory=list)
    split_train_ratio: float = 0.7
    split_val_ratio: float = 0.15
    split_test_ratio: float = 0.15
    stratify: bool = True
    group_aware: bool = True
    split_by: str = "source_video"
    class_balance_enabled: bool = True
    class_balance_method: str = "weighted_sampler"


@dataclass
class ImageDetectorConfig:
    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone_epochs: int = 5
    num_classes: int = 2
    dropout: float = 0.3
    embedding_dim: int = 512
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine_annealing"
    early_stopping_patience: int = 8
    mixed_precision: bool = False
    gradient_clip: float = 1.0
    loss_function: str = "cross_entropy"
    label_smoothing: float = 0.1


@dataclass
class VideoDetectorConfig:
    frame_backbone_architecture: str = "efficientnet_b0"
    frame_backbone_pretrained: bool = True
    frame_backbone_freeze_epochs: int = 5
    frame_embedding_dim: int = 512
    temporal_type: str = "lstm"
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 2
    temporal_dropout: float = 0.3
    temporal_bidirectional: bool = True
    num_classes: int = 2
    dropout: float = 0.4
    batch_size: int = 4
    epochs: int = 25
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine_annealing"
    early_stopping_patience: int = 8
    mixed_precision: bool = False
    gradient_clip: float = 1.0
    loss_function: str = "cross_entropy"
    label_smoothing: float = 0.1


@dataclass
class AudioDetectorConfig:
    architecture: str = "rawnet2"
    pretrained: bool = False
    num_classes: int = 2
    embedding_dim: int = 256
    sinc_out_channels: int = 70
    sinc_kernel_size: int = 1024
    resblock_channels: List[int] = field(default_factory=lambda: [70, 70, 140, 140, 280, 280])
    gru_hidden_dim: int = 1024
    gru_num_layers: int = 3
    fc_hidden_dim: int = 1024
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    early_stopping_patience: int = 15
    gradient_clip: float = 3.0
    loss_function: str = "amsoftmax"
    margin: float = 0.2
    scale: int = 30


@dataclass
class ExplainabilityConfig:
    image_method: str = "gradcam"
    image_target_layer: str = "features.7"
    image_alpha: float = 0.5
    image_colormap: str = "jet"
    video_method: str = "gradcam_per_frame"
    video_target_layer: str = "features.7"
    video_alpha: float = 0.5
    video_colormap: str = "jet"
    video_max_frames_visualize: int = 8
    audio_method: str = "integrated_gradients"
    audio_baseline: str = "zero"
    audio_steps: int = 50


@dataclass
class ForensicConfig:
    image_extract_exif: bool = True
    image_check_consistency: bool = True
    image_analyze_compression: bool = True
    image_detect_splicing: bool = True
    video_extract_metadata: bool = True
    video_check_codec: bool = True
    video_analyze_frame_consistency: bool = True
    video_detect_frame_duplication: bool = True
    video_analyze_gop_structure: bool = True
    audio_extract_metadata: bool = True
    audio_analyze_spectral_consistency: bool = True
    audio_detect_splicing: bool = True
    audio_check_sample_rate_consistency: bool = True


@dataclass
class RiskAssessmentConfig:
    deepfake_probability_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9
    })
    context_signals: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "impersonation": {"weight": 0.3, "keywords": ["official", "government", "police", "bank", "ceo", "director", "manager", "hr", "recruitment"]},
        "fraud": {"weight": 0.25, "keywords": ["urgent", "immediate", "transfer", "payment", "account", "otp", "kyc", "verify", "click here", "link"]},
        "harassment": {"weight": 0.2, "keywords": ["threat", "blackmail", "expose", "shame", "ruin", "destroy"]},
        "misinformation": {"weight": 0.15, "keywords": ["breaking", "exclusive", "leaked", "secret", "hidden truth", "they don't want you to know"]},
        "social_engineering": {"weight": 0.1, "keywords": ["trust", "confidential", "between us", "don't tell", "secret"]}
    })
    india_context_enabled: bool = True


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "eer", "confusion_matrix", "per_class_metrics"])
    robustness_enabled: bool = True
    robustness_transformations: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "jpeg_compression", "params": [75, 50, 30, 10]},
        {"name": "resize", "params": [0.5, 0.25]},
        {"name": "gaussian_noise", "params": [0.01, 0.05]},
        {"name": "blur", "params": [1.0, 2.0]},
        {"name": "frame_drop", "params": [0.1, 0.3]},
        {"name": "audio_compression", "params": ["mp3_128k", "mp3_64k"]}
    ])
    threshold_analysis_enabled: bool = True
    num_thresholds: int = 100
    plot_roc: bool = True
    plot_pr: bool = True
    plot_det: bool = True


@dataclass
class BackendConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"])
    database_url: str = "sqlite:///./deepfake_analysis.db"
    max_image_size_mb: int = 10
    max_video_size_mb: int = 100
    max_audio_size_mb: int = 20
    confidence_threshold: float = 0.5
    enable_explainability: bool = True
    enable_forensic: bool = True
    enable_risk_assessment: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/deepfake_detection.log"
    max_bytes: int = 10485760
    backup_count: int = 5
    console: bool = True


@dataclass
class MLConfig:
    general: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    image_detector: ImageDetectorConfig = field(default_factory=ImageDetectorConfig)
    video_detector: VideoDetectorConfig = field(default_factory=VideoDetectorConfig)
    audio_detector: AudioDetectorConfig = field(default_factory=AudioDetectorConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    forensic: ForensicConfig = field(default_factory=ForensicConfig)
    risk_assessment: RiskAssessmentConfig = field(default_factory=RiskAssessmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> MLConfig:
    """Load configuration from YAML file"""
    config = MLConfig()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        if yaml_config:
            # Update config from YAML
            if 'general' in yaml_config:
                config.general = yaml_config['general']
            if 'paths' in yaml_config:
                config.paths = yaml_config['paths']
            # Note: For nested dataclasses, manual mapping would be needed
            # This is a simplified version - in production use a proper config library
    
    return config


def get_default_config() -> MLConfig:
    """Get default configuration"""
    return MLConfig()


# Global config instance
_config: Optional[MLConfig] = None


def get_config() -> MLConfig:
    global _config
    if _config is None:
        _config = get_default_config()
    return _config


def set_config(config: MLConfig):
    global _config
    _config = config