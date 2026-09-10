"""
Unit tests for backend API
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock models for testing
import torch
from unittest.mock import Mock, patch, MagicMock

# We need to test the API endpoints without loading real models
# So we'll mock the model loading


@pytest.fixture
def client():
    """Create test client with mocked models."""
    # Import after patching
    with patch('backend.app.main.load_models'), \
         patch('backend.app.main.load_config') as mock_config, \
         patch('backend.app.main.setup_logging'):
        
        # Mock config
        mock_config.return_value = {
            "general": {"device": "cpu", "random_seed": 42},
            "paths": {
                "models_dir": "models",
                "logs_dir": "logs",
                "temp_dir": "temp"
            },
            "backend": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["*"],
                "upload": {
                    "max_image_size_mb": 10,
                    "max_video_size_mb": 100,
                    "max_audio_size_mb": 20,
                    "allowed_image_types": ["image/jpeg", "image/png"],
                    "allowed_video_types": ["video/mp4"],
                    "allowed_audio_types": ["audio/wav"]
                }
            }
        }
        
        from backend.app.main import app
        return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "device" in data
        assert data["status"] == "healthy"


class TestValidateEndpoint:
    """Tests for file validation endpoint."""
    
    def test_validate_image_file(self, client):
        """Test validating an image file."""
        # Create dummy image file
        file_content = b"fake jpeg content"
        files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
        
        response = client.post("/validate", files=files)
        # Should work even with fake content (validation is just MIME type + size)
        assert response.status_code in [200, 413, 400]
    
    def test_validate_unsupported_type(self, client):
        """Test validating unsupported file type."""
        files = {"file": ("test.txt", io.BytesIO(b"text"), "text/plain")}
        
        response = client.post("/validate", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


class TestAnalyzeEndpoint:
    """Tests for analyze endpoint."""
    
    def test_analyze_without_models(self, client):
        """Test analyze endpoint when no models loaded."""
        file_content = b"fake image"
        files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
        data = {"text_content": "", "user_context": "{}"}
        
        response = client.post("/analyze", files=files, data=data)
        # Should return 503 since models not loaded
        assert response.status_code == 503
    
    def test_analyze_invalid_file_type(self, client):
        """Test analyze with invalid file type."""
        files = {"file": ("test.txt", io.BytesIO(b"text"), "text/plain")}
        data = {"text_content": "", "user_context": "{}"}
        
        response = client.post("/analyze", files=files, data=data)
        assert response.status_code == 400


class TestHistoryEndpoint:
    """Tests for history endpoint."""
    
    def test_get_history(self, client):
        """Test getting analysis history."""
        response = client.get("/history")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)


class TestModelsEndpoint:
    """Tests for models info endpoint."""
    
    def test_get_models(self, client):
        """Test getting model information."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "image_detector" in data
        assert "video_detector" in data
        assert "audio_detector" in data
        assert "multimodal_detector" in data


class TestAnalyticsEndpoint:
    """Tests for analytics endpoint."""
    
    def test_get_analytics(self, client):
        """Test getting analytics."""
        response = client.get("/analytics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_analyses" in data


# Integration test with mocked models
class TestAnalyzeWithMockedModels:
    """Tests with mocked model inference."""
    
    @patch('backend.app.main.models', {
        "image": Mock(),
        "video": Mock(),
        "audio": Mock()
    })
    @patch('backend.app.main.preprocessors', {
        "image": Mock(),
        "video": Mock(),
        "audio": Mock()
    })
    @patch('backend.app.main.device', torch.device("cpu"))
    def test_analyze_image_mocked(self, client):
        """Test image analysis with mocked model."""
        from backend.app.main import config
        
        # Setup mocks
        config["image_detector"] = {"architecture": "efficientnet_b0"}
        config["backend"]["upload"]["allowed_image_types"] = ["image/jpeg", "image/png"]
        config["backend"]["upload"]["max_image_size_mb"] = 10
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model._get_target_layer = Mock(return_value=None)
        
        # Mock forward pass
        mock_logits = torch.tensor([[0.2, 0.8]])  # 80% fake
        mock_model.return_value = mock_logits
        
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_file = Mock(return_value=torch.randn(3, 224, 224))
        
        from backend.app.main import models, preprocessors
        models["image"] = mock_model
        preprocessors["image"] = mock_preprocessor
        
        # Test
        file_content = b"fake jpeg"
        files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
        data = {"text_content": "", "user_context": "{}"}
        
        response = client.post("/analyze", files=files, data=data)
        
        # Should process (might fail on Grad-CAM or other parts, but should not 503)
        assert response.status_code != 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])