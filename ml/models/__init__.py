"""
Model architectures for deepfake detection.

Canonical implementations (torchvision/RawNet2 based, integrated with
training scripts and backend inference):
- image_detector: DeepfakeImageDetector (+ ensemble)
- video_detector: DeepfakeVideoDetector (+ temporal aggregators)
- audio_detector: RawNet2 / Wav2Vec2 / Conformer / AST
- multimodal_detector: MultimodalDeepfakeDetector (+ fusion modules)
"""
from .image_detector import (
    DeepfakeImageDetector,
    EnsembleImageDetector,
    create_image_detector,
    load_image_detector,
    save_image_detector,
    get_model_metadata,
)
from .video_detector import (
    DeepfakeVideoDetector,
    TemporalAggregator,
    MeanAggregator,
    MaxAggregator,
    AttentionAggregator,
    LSTMAggregator,
    TransformerAggregator,
    create_temporal_aggregator,
    create_video_detector,
    load_video_detector,
    save_video_detector,
)
from .audio_detector import (
    RawNet2,
    AMSoftmax,
    Wav2Vec2Detector,
    ConformerDetector,
    ASTDetector,
    create_audio_detector,
    load_audio_detector,
    save_audio_detector,
)
from .multimodal_detector import (
    MultimodalDeepfakeDetector,
    FusionModule,
    ConcatFusion,
    AttentionFusion,
    BilinearFusion,
    GatedFusion,
    create_fusion_module,
    create_multimodal_detector,
    load_multimodal_detector,
    save_multimodal_detector,
)

__all__ = [
    "DeepfakeImageDetector",
    "EnsembleImageDetector",
    "create_image_detector",
    "load_image_detector",
    "save_image_detector",
    "get_model_metadata",
    "DeepfakeVideoDetector",
    "TemporalAggregator",
    "MeanAggregator",
    "MaxAggregator",
    "AttentionAggregator",
    "LSTMAggregator",
    "TransformerAggregator",
    "create_temporal_aggregator",
    "create_video_detector",
    "load_video_detector",
    "save_video_detector",
    "RawNet2",
    "AMSoftmax",
    "Wav2Vec2Detector",
    "ConformerDetector",
    "ASTDetector",
    "create_audio_detector",
    "load_audio_detector",
    "save_audio_detector",
    "MultimodalDeepfakeDetector",
    "FusionModule",
    "ConcatFusion",
    "AttentionFusion",
    "BilinearFusion",
    "GatedFusion",
    "create_fusion_module",
    "create_multimodal_detector",
    "load_multimodal_detector",
    "save_multimodal_detector",
]
