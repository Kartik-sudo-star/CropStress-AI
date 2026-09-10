"""
Unit tests for ML models
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.image_detector import DeepfakeImageDetector, create_image_detector
from ml.models.video_detector import DeepfakeVideoDetector, create_video_detector, create_temporal_aggregator
from ml.models.audio_detector import RawNet2, create_audio_detector
from ml.models.multimodal_detector import MultimodalDeepfakeDetector, create_multimodal_detector, create_fusion_module


class TestImageDetector:
    """Tests for image detector model."""
    
    def test_model_creation_efficientnet(self):
        """Test creating EfficientNet-B0 detector."""
        model = DeepfakeImageDetector(
            architecture="efficientnet_b0",
            pretrained=False,
            num_classes=2,
            embedding_dim=512
        )
        
        assert model.architecture == "efficientnet_b0"
        assert model.num_classes == 2
    
    def test_model_creation_resnet(self):
        """Test creating ResNet50 detector."""
        model = DeepfakeImageDetector(
            architecture="resnet50",
            pretrained=False,
            num_classes=2
        )
        
        assert model.architecture == "resnet50"
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DeepfakeImageDetector(
            architecture="efficientnet_b0",
            pretrained=False,
            num_classes=2
        )
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (2, 2)
        assert not torch.isnan(logits).any()
    
    def test_get_embeddings(self):
        """Test getting embeddings."""
        model = DeepfakeImageDetector(
            architecture="efficientnet_b0",
            pretrained=False,
            num_classes=2,
            embedding_dim=256
        )
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            embeddings = model.get_embeddings(x)
        
        assert embeddings.shape == (1, 256)
    
    def test_freeze_unfreeze(self):
        """Test freeze/unfreeze backbone."""
        model = DeepfakeImageDetector(
            architecture="efficientnet_b0",
            pretrained=False,
            freeze_backbone_epochs=5
        )
        
        # Initially unfrozen
        model.set_epoch(0)
        for param in model.backbone.parameters():
            assert not param.requires_grad  # Should be frozen at epoch 0
        
        model.set_epoch(10)
        for param in model.backbone.parameters():
            assert param.requires_grad  # Should be unfrozen after epoch 5
    
    def test_create_from_config(self):
        """Test creating model from config dict."""
        config = {
            "image_detector": {
                "architecture": "efficientnet_b0",
                "pretrained": False,
                "freeze_backbone_epochs": 5,
                "num_classes": 2,
                "dropout": 0.3,
                "embedding_dim": 512
            }
        }
        
        model = create_image_detector(config)
        assert isinstance(model, DeepfakeImageDetector)
        assert model.architecture == "efficientnet_b0"


class TestVideoDetector:
    """Tests for video detector model."""
    
    def test_temporal_aggregators(self):
        """Test different temporal aggregators."""
        aggregator_types = ["mean", "max", "attention", "lstm", "transformer"]
        
        for agg_type in aggregator_types:
            agg = create_temporal_aggregator(
                agg_type, input_dim=512, hidden_dim=256, dropout=0.3
            )
            
            x = torch.randn(2, 16, 512)  # [B, T, D]
            out = agg(x)
            
            assert out.shape == (2, 512), f"{agg_type} output shape mismatch"
    
    def test_video_detector_creation(self):
        """Test creating video detector."""
        model = DeepfakeVideoDetector(
            frame_backbone="efficientnet_b0",
            pretrained=False,
            temporal_type="lstm",
            num_classes=2
        )
        
        assert model.frame_backbone_name == "efficientnet_b0"
    
    def test_video_forward_pass(self):
        """Test video forward pass."""
        model = DeepfakeVideoDetector(
            frame_backbone="efficientnet_b0",
            pretrained=False,
            temporal_type="mean",  # Simplest for testing
            num_classes=2
        )
        model.eval()
        
        # [B, T, C, H, W]
        x = torch.randn(1, 8, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 2)
    
    def test_per_frame_logits(self):
        """Test per-frame predictions."""
        model = DeepfakeVideoDetector(
            frame_backbone="efficientnet_b0",
            pretrained=False,
            temporal_type="mean",
            num_classes=2
        )
        model.eval()
        
        x = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            frame_logits = model.get_per_frame_logits(x)
        
        assert frame_logits.shape == (1, 4, 2)


class TestAudioDetector:
    """Tests for audio detector model."""
    
    def test_rawnet2_creation(self):
        """Test creating RawNet2."""
        model = RawNet2(
            num_classes=2,
            embedding_dim=256
        )
        
        assert model.num_classes == 2
        assert model.embedding_dim == 256
    
    def test_rawnet2_forward(self):
        """Test RawNet2 forward pass."""
        model = RawNet2(
            sinc_conv={"out_channels": 32, "kernel_size": 256},  # Smaller for testing
            resblock={"channels": [32, 32, 64]},
            gru={"hidden_dim": 128, "num_layers": 1},
            fc={"hidden_dim": 128},
            num_classes=2,
            embedding_dim=128
        )
        model.eval()
        
        # RawNet2 expects [B, 1, T] or [B, T]
        x = torch.randn(1, 1, 16000)  # 1 second at 16kHz
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 2)
    
    def test_get_embeddings(self):
        """Test getting audio embeddings."""
        model = RawNet2(
            sinc_conv={"out_channels": 32, "kernel_size": 256},
            resblock={"channels": [32, 32]},
            gru={"hidden_dim": 64, "num_layers": 1},
            fc={"hidden_dim": 64},
            num_classes=2,
            embedding_dim=64
        )
        model.eval()
        
        x = torch.randn(1, 1, 16000)
        with torch.no_grad():
            emb = model.get_embeddings(x)
        
        assert emb.shape == (1, 64)


class TestMultimodalDetector:
    """Tests for multimodal detector."""
    
    def test_fusion_modules(self):
        """Test different fusion modules."""
        fusion_types = ["concatenate", "attention", "bilinear", "gated"]
        
        for fusion_type in fusion_types:
            fusion = create_fusion_module(
                fusion_type, visual_dim=512, audio_dim=256, output_dim=256
            )
            
            v = torch.randn(2, 512)
            a = torch.randn(2, 256)
            out = fusion(v, a)
            
            assert out.shape == (2, 256), f"{fusion_type} output shape mismatch"
    
    def test_multimodal_creation(self):
        """Test creating multimodal detector."""
        model = MultimodalDeepfakeDetector(
            visual_branch="image",
            visual_backbone="efficientnet_b0",
            audio_architecture="rawnet2",
            audio_config={"rawnet2": {"sinc_conv": {"out_channels": 32, "kernel_size": 256}}},
            num_classes=2
        )
        
        assert model.visual_branch == "image"
        assert model.audio_architecture == "rawnet2"
    
    def test_multimodal_forward(self):
        """Test multimodal forward pass (image + audio)."""
        model = MultimodalDeepfakeDetector(
            visual_branch="image",
            visual_backbone="efficientnet_b0",
            pretrained=False,
            audio_architecture="rawnet2",
            audio_config={
                "rawnet2": {
                    "sinc_conv": {"out_channels": 32, "kernel_size": 256},
                    "resblock": {"channels": [32, 32]},
                    "gru": {"hidden_dim": 64, "num_layers": 1},
                    "fc": {"hidden_dim": 64}
                }
            },
            audio_embedding_dim=64,
            fusion_type="concatenate",
            num_classes=2
        )
        model.eval()
        
        # Image input: [B, C, H, W]
        visual = torch.randn(1, 3, 224, 224)
        # Audio input: [B, 1, T]
        audio = torch.randn(1, 1, 16000)
        
        with torch.no_grad():
            logits = model(visual, audio)
        
        assert logits.shape == (1, 2)


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_image_detector_training_step(self):
        """Test a single training step for image detector."""
        model = DeepfakeImageDetector(
            architecture="efficientnet_b0",
            pretrained=False,
            num_classes=2
        )
        model.train()
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 2, (4,))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_video_detector_training_step(self):
        """Test a single training step for video detector."""
        model = DeepfakeVideoDetector(
            frame_backbone="efficientnet_b0",
            pretrained=False,
            temporal_type="mean",
            num_classes=2
        )
        model.train()
        
        x = torch.randn(2, 4, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])