"""
Training Package
"""
# Expose training functions
from ml.training.train_image import main as train_image_main
from ml.training.train_video import main as train_video_main
from ml.training.train_audio import main as train_audio_main
from ml.training.train_multimodal import main as train_multimodal_main

__all__ = [
    "train_image_main",
    "train_video_main",
    "train_audio_main",
    "train_multimodal_main",
]