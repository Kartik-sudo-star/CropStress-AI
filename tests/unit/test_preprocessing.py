"""
Unit tests for preprocessing modules
"""
import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

# Add project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.preprocessing.image_preprocessing import ImagePreprocessor, FaceDetector, load_image, validate_image
from ml.preprocessing.video_preprocessing import VideoPreprocessor, FrameSampler, VideoMetadataExtractor
from ml.preprocessing.audio_preprocessing import AudioPreprocessor, AudioMetadataExtractor


class TestImagePreprocessing:
    """Tests for image preprocessing."""
    
    def test_image_preprocessor_creation(self):
        """Test creating image preprocessor."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.mean == [0.485, 0.456, 0.406]
    
    def test_preprocess_pil_image(self):
        """Test preprocessing a PIL image."""
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # Create dummy image
        img = Image.new('RGB', (300, 300), color='red')
        
        # Preprocess
        tensor = preprocessor.preprocess(img, is_training=False)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_preprocess_training_mode(self):
        """Test preprocessing with augmentations."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            augmentation_config={"horizontal_flip": 1.0}  # Always flip
        )
        
        img = Image.new('RGB', (300, 300), color='blue')
        
        # Training mode should apply augmentations
        tensor_train = preprocessor.preprocess(img, is_training=True)
        tensor_val = preprocessor.preprocess(img, is_training=False)
        
        # They should be different due to augmentation
        assert not torch.allclose(tensor_train, tensor_val)
    
    def test_preprocess_file(self):
        """Test preprocessing from file."""
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # Create temp image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (300, 300), color='green')
            img.save(f, 'JPEG')
            temp_path = f.name
        
        try:
            tensor = preprocessor.preprocess_file(temp_path)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (3, 224, 224)
        finally:
            os.unlink(temp_path)
    
    def test_face_detector(self):
        """Test face detector (Haar cascade fallback)."""
        detector = FaceDetector(min_face_size=32, confidence_threshold=0.5)
        
        # Create image with face-like pattern
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Simple pattern that might trigger Haar
        cv2_rect = cv2.rectangle(img.copy(), (50, 50), (150, 150), (255, 255, 255), -1)
        
        faces = detector.detect(cv2_rect)
        # May or may not detect - just ensure it runs
        assert isinstance(faces, list)
    
    def test_load_image(self):
        """Test image loading utility."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(f, 'PNG')
            temp_path = f.name
        
        try:
            loaded = load_image(temp_path)
            assert isinstance(loaded, Image.Image)
            assert loaded.mode == 'RGB'
            assert loaded.size == (100, 100)
        finally:
            os.unlink(temp_path)
    
    def test_validate_image(self):
        """Test image validation."""
        img = Image.new('RGB', (100, 100), color='red')
        valid, msg = validate_image(img, min_size=(32, 32))
        assert valid
        assert msg == "OK"
        
        # Too small
        img_small = Image.new('RGB', (16, 16), color='red')
        valid, msg = validate_image(img_small, min_size=(32, 32))
        assert not valid
        assert "too small" in msg.lower()


class TestVideoPreprocessing:
    """Tests for video preprocessing."""
    
    def test_frame_sampler_uniform(self):
        """Test uniform frame sampling."""
        sampler = FrameSampler(method="uniform", num_frames=16, min_frames=8)
        indices = sampler.sample_indices(100, 30.0)
        
        assert len(indices) == 16
        assert indices[0] == 0
        assert indices[-1] == 99
        assert all(indices[i] < indices[i+1] for i in range(len(indices)-1))
    
    def test_frame_sampler_fewer_frames(self):
        """Test sampling when video has fewer frames than requested."""
        sampler = FrameSampler(method="uniform", num_frames=16, min_frames=8)
        indices = sampler.sample_indices(10, 30.0)
        
        assert len(indices) == 10  # Returns all frames
        assert indices == list(range(10))
    
    def test_video_metadata_extractor(self):
        """Test video metadata extraction (requires test video)."""
        # This would need a real video file
        pass


class TestAudioPreprocessing:
    """Tests for audio preprocessing."""
    
    def test_audio_preprocessor_creation(self):
        """Test creating audio preprocessor."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            segment_length=4.0,
            spectrogram_config={"n_mels": 80, "target_length": 400}
        )
        
        assert preprocessor.sample_rate == 16000
        assert preprocessor.segment_length == 4.0
        assert preprocessor.segment_samples == 64000
    
    def test_audio_metadata_extractor(self):
        """Test audio metadata extraction."""
        # Would need a real audio file
        pass


# Integration test
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_image_preprocessing_pipeline(self):
        """Test full image preprocessing pipeline."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            face_detection_config={"enabled": False}  # Disable for speed
        )
        
        # Batch processing
        images = [Image.new('RGB', (300, 300), color=c) for c in ['red', 'green', 'blue']]
        batch = preprocessor.preprocess_batch(images, is_training=False)
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (3, 3, 224, 224)
    
    def test_preprocessor_save_load(self):
        """Test saving and loading preprocessor config."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            augmentation_config={"horizontal_flip": 0.5}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            preprocessor.save(temp_path)
            loaded = ImagePreprocessor.load(temp_path)
            
            assert loaded.target_size == preprocessor.target_size
            assert loaded.augmentation_config == preprocessor.augmentation_config
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])