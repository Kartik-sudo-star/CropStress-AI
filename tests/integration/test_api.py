"""
Integration Tests for Backend API
"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (224, 224), color='green')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


@pytest.fixture
def valid_sensor_data():
    """Valid sensor data for testing."""
    return {
        'soil_moisture': 45.0,
        'temperature': 24.5,
        'humidity': 65.0,
        'rainfall': 12.3,
        'light_intensity': 850,
    }


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'version' in data
        assert 'models_loaded' in data
        assert 'database' in data
    
    def test_readiness_check(self, client):
        """Test readiness endpoint."""
        response = client.get('/api/health/ready')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
    
    def test_liveness_check(self, client):
        """Test liveness endpoint."""
        response = client.get('/api/health/live')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'alive'


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    @patch('backend.app.services.model_loader.get_models')
    def test_predict_multimodal_success(self, mock_get_models, client, sample_image, valid_sensor_data):
        """Test successful multimodal prediction."""
        # Mock models
        mock_models = {
            'multimodal': MagicMock(),
            'image_only': MagicMock(),
            'sensor_only': MagicMock(),
        }
        mock_get_models.return_value = mock_models
        
        # Mock prediction service
        with patch('backend.app.api.prediction.predict_multimodal') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Moderate Stress',
                'confidence': 0.87,
                'probabilities': {
                    'Healthy': 0.05,
                    'Mild Stress': 0.12,
                    'Moderate Stress': 0.87,
                    'Severe Stress': 0.06,
                },
                'model_version': 'v1.0.0',
                'gradcam_base64': 'fake_base64',
                'shap_values': {'soil_moisture': 0.34, 'temperature': 0.28},
            }
            
            # Prepare form data
            files = {'image': ('test.jpg', sample_image.getvalue(), 'image/jpeg')}
            data = {**valid_sensor_data, 'crop': 'wheat'}
            
            response = client.post('/api/predict', files=files, data=data)
            
            assert response.status_code == 200
            result = response.json()
            assert 'prediction_id' in result
            assert result['prediction'] == 'Moderate Stress'
            assert result['confidence'] == 0.87
            assert 'probabilities' in result
            assert 'contributing_factors' in result
            assert 'recommendations' in result
    
    def test_predict_multimodal_invalid_image(self, client, valid_sensor_data):
        """Test prediction with invalid image type."""
        # Create a text file instead of image
        files = {'image': ('test.txt', b'not an image', 'text/plain')}
        data = valid_sensor_data
        
        response = client.post('/api/predict', files=files, data=data)
        
        assert response.status_code == 400
        assert 'Invalid image type' in response.json()['detail']
    
    def test_predict_multimodal_missing_sensor(self, client, sample_image):
        """Test prediction with missing sensor data."""
        files = {'image': ('test.jpg', sample_image.getvalue(), 'image/jpeg')}
        data = {'soil_moisture': 45.0}  # Missing other fields
        
        response = client.post('/api/predict', files=files, data=data)
        
        assert response.status_code == 400
    
    @patch('backend.app.services.model_loader.get_models')
    def test_predict_image_only(self, mock_get_models, client, sample_image):
        """Test image-only prediction."""
        mock_models = {
            'image_only': MagicMock(),
        }
        mock_get_models.return_value = mock_models
        
        with patch('backend.app.api.prediction.predict_image_only') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Mild Stress',
                'confidence': 0.75,
                'probabilities': {
                    'Healthy': 0.15,
                    'Mild Stress': 0.75,
                    'Moderate Stress': 0.08,
                    'Severe Stress': 0.02,
                },
                'model_version': 'v1.0.0',
            }
            
            files = {'image': ('test.jpg', sample_image.getvalue(), 'image/jpeg')}
            data = {'crop': 'rice'}
            
            response = client.post('/api/predict/image', files=files, data=data)
            
            assert response.status_code == 200
            result = response.json()
            assert result['prediction'] == 'Mild Stress'
            assert result['model_type'] == 'image_only'
    
    @patch('backend.app.services.model_loader.get_models')
    def test_predict_sensor_only(self, mock_get_models, client, valid_sensor_data):
        """Test sensor-only prediction."""
        mock_models = {
            'sensor_only': MagicMock(),
        }
        mock_get_models.return_value = mock_models
        
        with patch('backend.app.api.prediction.predict_sensor_only') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Healthy',
                'confidence': 0.92,
                'probabilities': {
                    'Healthy': 0.92,
                    'Mild Stress': 0.05,
                    'Moderate Stress': 0.02,
                    'Severe Stress': 0.01,
                },
                'model_version': 'v1.0.0',
            }
            
            response = client.post('/api/predict/sensor', json={
                'crop': 'maize',
                'sensor_data': valid_sensor_data,
            })
            
            assert response.status_code == 200
            result = response.json()
            assert result['prediction'] == 'Healthy'
            assert result['model_type'] == 'sensor_only'
    
    def test_predict_sensor_only_invalid_data(self, client):
        """Test sensor-only prediction with invalid data."""
        invalid_data = {
            'sensor_data': {
                'soil_moisture': 150,  # Out of range
                'temperature': 25,
                'humidity': 60,
                'rainfall': 10,
                'light_intensity': 800,
            }
        }
        
        response = client.post('/api/predict/sensor', json=invalid_data)
        
        assert response.status_code == 400
        assert 'outside valid range' in response.json()['detail']


class TestModelEndpoints:
    """Tests for model comparison endpoints."""
    
    def test_get_model_comparison(self, client):
        """Test model comparison endpoint."""
        response = client.get('/api/models/comparison')
        assert response.status_code == 200
        data = response.json()
        # Should have at least some model types
        assert isinstance(data, dict)
    
    def test_get_model_versions(self, client):
        """Test model versions endpoint."""
        response = client.get('/api/models/versions')
        assert response.status_code == 200
        data = response.json()
        assert 'versions' in data
        assert 'loaded' in data


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""
    
    def test_get_analytics(self, client):
        """Test analytics endpoint."""
        response = client.get('/api/analytics')
        assert response.status_code == 200
        data = response.json()
        assert 'period' in data
        assert 'summary' in data
    
    def test_get_feature_importance(self, client):
        """Test feature importance endpoint."""
        response = client.get('/api/analytics/feature-importance')
        assert response.status_code == 200
        data = response.json()
        assert 'feature_importance' in data
    
    def test_get_confusion_matrix(self, client):
        """Test confusion matrix endpoint."""
        response = client.get('/api/analytics/confusion-matrix')
        assert response.status_code == 200
        data = response.json()
        assert 'confusion_matrices' in data


class TestHistoryEndpoints:
    """Tests for history endpoints."""
    
    def test_get_predictions(self, client):
        """Test get predictions list."""
        response = client.get('/api/predictions')
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert 'total' in data
        assert 'page' in data
        assert 'total_pages' in data
    
    def test_get_predictions_with_filters(self, client):
        """Test get predictions with query parameters."""
        response = client.get('/api/predictions?page=1&limit=5&risk=high')
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
    
    def test_get_prediction_by_id(self, client):
        """Test get single prediction."""
        # First get list to find an ID
        list_response = client.get('/api/predictions?limit=1')
        if list_response.json()['total'] > 0:
            pred_id = list_response.json()['predictions'][0]['prediction_id']
            response = client.get(f'/api/predictions/{pred_id}')
            assert response.status_code == 200
            data = response.json()
            assert data['prediction_id'] == pred_id
        else:
            # No predictions yet, test 404
            response = client.get('/api/predictions/nonexistent')
            assert response.status_code == 404
    
    def test_get_history_summary(self, client):
        """Test history summary endpoint."""
        response = client.get('/api/predictions/stats/summary?days=7')
        assert response.status_code == 200
        data = response.json()
        assert 'period_days' in data
        assert 'total_predictions' in data


class TestValidationEndpoint:
    """Tests for input validation endpoint."""
    
    def test_validate_valid_input(self, client, valid_sensor_data):
        """Test validation of valid input."""
        response = client.post('/api/validate-input', json={
            'sensor_data': valid_sensor_data,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['valid'] == True
        assert len(data['errors']) == 0
    
    def test_validate_invalid_input(self, client):
        """Test validation of invalid input."""
        invalid_data = {
            'sensor_data': {
                'soil_moisture': 150,  # Invalid
                'temperature': 25,
                'humidity': 60,
                'rainfall': 10,
                'light_intensity': 800,
            }
        }
        
        response = client.post('/api/validate-input', json=invalid_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['valid'] == False
        assert len(data['errors']) > 0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])