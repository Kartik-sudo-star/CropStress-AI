"""
Integration tests for the full pipeline
"""
import pytest
import torch
import tempfile
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.preprocessing import ImagePreprocessor, VideoPreprocessor, AudioPreprocessor
from ml.models import (
    DeepfakeImageDetector, DeepfakeVideoDetector,
    RawNet2, MultimodalDeepfakeDetector
)
from ml.evidence import create_evidence_manager, AnalysisRecord
from ml.risk import create_risk_assessor, RiskLevel
from ml.forensic import create_forensic_analyzer


class TestFullImagePipeline:
    """Integration test for complete image analysis pipeline."""
    
    def test_image_preprocess_to_detection(self):
        """Test image preprocessing -> model -> evidence."""
        # Create preprocessor
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            face_detection_config={"enabled": False}
        )
        
        # Create model
        model = DeepfakeImageDetector(
            architecture="efficientnet_b0",
            pretrained=False,
            num_classes=2
        )
        model.eval()
        
        # Create dummy image
        from PIL import Image
        img = Image.new('RGB', (300, 300), color='red')
        
        # Preprocess
        tensor = preprocessor.preprocess(img, is_training=False)
        assert tensor.shape == (3, 224, 224)
        
        # Inference
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()
        
        assert pred in [0, 1]
        assert 0 <= confidence <= 1
        
        # Create evidence record
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f, 'JPEG')
            temp_path = f.name
        
        try:
            evidence_mgr = create_evidence_manager({})
            record = evidence_mgr.create_analysis_record(
                file_path=temp_path,
                media_type="image",
                media_properties={"width": 300, "height": 300},
                model_version="test-1.0",
                prediction="Potentially Manipulated" if pred == 1 else "Likely Authentic",
                confidence=confidence,
                risk_level="MEDIUM",
                forensic_indicators={},
                risk_assessment={"risk_score": 0.5},
                explanation_artifacts={},
                recommendations=["Test recommendation"]
            )
            
            assert record.analysis_id.startswith("ANL-")
            assert record.file_hash
            assert len(record.file_hash) == 64  # SHA-256
        finally:
            import os
            os.unlink(temp_path)


class TestVideoPipeline:
    """Integration test for video pipeline (simplified)."""
    
    def test_video_preprocessor_creation(self):
        """Test video preprocessor creation."""
        preprocessor = VideoPreprocessor(
            target_size=(224, 224),
            frame_sampling_config={"method": "uniform", "num_frames": 8},
            face_detection_config={"enabled": False}
        )
        
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.frame_sampler.num_frames == 8
    
    def test_video_model_forward(self):
        """Test video model forward pass."""
        model = DeepfakeVideoDetector(
            frame_backbone="efficientnet_b0",
            pretrained=False,
            temporal_type="mean",
            num_classes=2
        )
        model.eval()
        
        # Dummy video tensor [B, T, C, H, W]
        x = torch.randn(1, 4, 3, 224, 224)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 2)


class TestAudioPipeline:
    """Integration test for audio pipeline."""
    
    def test_audio_preprocessor_creation(self):
        """Test audio preprocessor creation."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            segment_length=4.0,
            spectrogram_config={"n_mels": 80, "target_length": 400}
        )
        
        assert preprocessor.sample_rate == 16000
        assert preprocessor.segment_samples == 64000
    
    def test_audio_model_forward(self):
        """Test audio model forward pass."""
        model = RawNet2(
            sinc_conv={"out_channels": 32, "kernel_size": 256},
            resblock={"channels": [32, 32]},
            gru={"hidden_dim": 64, "num_layers": 1},
            fc={"hidden_dim": 64},
            num_classes=2,
            embedding_dim=64
        )
        model.eval()
        
        # RawNet2 expects waveform
        x = torch.randn(1, 1, 16000)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 2)


class TestMultimodalPipeline:
    """Integration test for multimodal pipeline."""
    
    def test_multimodal_model_creation(self):
        """Test multimodal model creation."""
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
        
        assert model.visual_branch == "image"
        assert model.audio_architecture == "rawnet2"
    
    def test_multimodal_forward(self):
        """Test multimodal forward pass."""
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
        
        visual = torch.randn(1, 3, 224, 224)
        audio = torch.randn(1, 1, 16000)
        
        with torch.no_grad():
            logits = model(visual, audio)
        
        assert logits.shape == (1, 2)


class TestEvidencePipeline:
    """Integration test for evidence and reporting pipeline."""
    
    def test_evidence_record_creation_and_retrieval(self):
        """Test creating and saving evidence record."""
        evidence_mgr = create_evidence_manager({
            "paths": {"reports_dir": "test_reports"}
        })
        
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f, 'PNG')
            temp_path = f.name
        
        try:
            record = evidence_mgr.create_analysis_record(
                file_path=temp_path,
                media_type="image",
                media_properties={"width": 100, "height": 100},
                model_version="test-1.0",
                prediction="Likely Authentic",
                confidence=0.85,
                risk_level="LOW",
                forensic_indicators={"test": "data"},
                risk_assessment={"risk_score": 0.2},
                explanation_artifacts={},
                recommendations=["No action needed"]
            )
            
            # Save
            saved_path = evidence_mgr.save_record(record)
            assert saved_path.exists()
            
            # Load
            loaded = evidence_mgr.load_record(record.analysis_id)
            assert loaded is not None
            assert loaded.analysis_id == record.analysis_id
            assert loaded.prediction == "Likely Authentic"
            
            # Generate human report
            report_path = evidence_mgr.save_human_report(record)
            assert report_path.exists()
            
            # Check report content
            with open(report_path) as f:
                content = f.read()
            assert record.analysis_id in content
            assert "Likely Authentic" in content
            
        finally:
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
            # Cleanup test reports
            import shutil
            try:
                shutil.rmtree("test_reports")
            except:
                pass


class TestRiskAssessmentPipeline:
    """Integration test for risk assessment pipeline."""
    
    def test_risk_assessment_full(self):
        """Test complete risk assessment."""
        config = {
            "risk_assessment": {
                "thresholds": {
                    "deepfake_probability": {
                        "low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9
                    }
                },
                "context_signals": {
                    "impersonation_indicators": {
                        "weight": 0.3,
                        "keywords": ["police", "bank", "government"]
                    },
                    "fraud_indicators": {
                        "weight": 0.25,
                        "keywords": ["urgent", "payment", "transfer"]
                    }
                },
                "india_context": {"enabled": True}
            }
        }
        
        assessor = create_risk_assessor(config)
        
        # Test with fraud keywords
        result = assessor.assess(
            deepfake_probability=0.8,
            media_type="image",
            text_content="Urgent payment required, transfer money immediately",
            metadata={"software": "Photoshop"},
            forensic_indicators={"ela": {"suspicious_ratio": 0.15}},
            user_context={"platform": "whatsapp"}
        )
        
        assert isinstance(result.risk_level, RiskLevel)
        assert 0 <= result.risk_score <= 1
        assert result.deepfake_probability == 0.8
        assert len(result.indicators) > 0
        assert len(result.recommended_actions) > 0
        assert "india_specific_guidance" in result.india_specific_guidance


class TestForensicPipeline:
    """Integration test for forensic analysis."""
    
    def test_forensic_analyzer_image(self):
        """Test forensic analyzer on image."""
        from PIL import Image
        import tempfile
        
        analyzer = create_forensic_analyzer({})
        
        img = Image.new('RGB', (200, 200), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f, 'JPEG')
            temp_path = f.name
        
        try:
            results = analyzer.analyze_file(temp_path)
            
            assert results["file_type"] == "image"
            assert results["sha256"]
            assert len(results["sha256"]) == 64
            assert "metadata" in results
            
        finally:
            import os
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])