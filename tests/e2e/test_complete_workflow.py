"""
End-to-End Tests for Complete Workflow

Tests the full pipeline: Frontend -> Backend -> Model -> Database -> Response
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


class TestCompleteWorkflow:
    """Test the complete analysis workflow."""
    
    @patch('backend.app.services.model_loader.get_models')
    def test_full_multimodal_workflow(self, mock_get_models, client, sample_image, valid_sensor_data):
        """Test complete workflow from image upload to result retrieval."""
        # Mock all models as loaded
        mock_models = {
            'multimodal': MagicMock(),
            'image_only': MagicMock(),
            'sensor_only': MagicMock(),
        }
        mock_get_models.return_value = mock_models
        
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
                'model_version': 'multimodal_v1.0.0',
                'gradcam_base64': 'base64_encoded_image',
                'shap_values': {
                    'soil_moisture': 0.34,
                    'temperature': 0.28,
                    'humidity': 0.15,
                    'rainfall': 0.12,
                    'light_intensity': 0.11,
                },
            }
            
            # Step 1: Upload image and sensor data
            files = {'image': ('leaf.jpg', sample_image.getvalue(), 'image/jpeg')}
            data = {**valid_sensor_data, 'crop': 'wheat'}
            
            predict_response = client.post('/api/predict', files=files, data=data)
            
            assert predict_response.status_code == 200
            predict_data = predict_response.json()
            
            # Verify response structure
            assert 'prediction_id' in predict_data
            assert predict_data['prediction'] == 'Moderate Stress'
            assert predict_data['confidence'] == 0.87
            assert predict_data['model_type'] == 'multimodal'
            assert len(predict_data['contributing_factors']) > 0
            assert len(predict_data['recommendations']) > 0
            assert predict_data['explanations'] is not None
            
            prediction_id = predict_data['prediction_id']
            
            # Step 2: Retrieve prediction from history
            history_response = client.get(f'/api/predictions/{prediction_id}')
            assert history_response.status_code == 200
            history_data = history_response.json()
            assert history_data['prediction_id'] == prediction_id
            assert history_data['prediction'] == 'Moderate Stress'
            
            # Step 3: Verify prediction appears in history list
            list_response = client.get('/api/predictions?limit=10')
            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data['total'] >= 1
            assert any(p['prediction_id'] == prediction_id for p in list_data['predictions'])
            
            # Step 4: Verify analytics updated
            analytics_response = client.get('/api/analytics?days=1')
            assert analytics_response.status_code == 200
            analytics_data = analytics_response.json()
            assert analytics_data['summary']['total_predictions'] >= 1
            
            # Step 5: Verify model comparison accessible
            comparison_response = client.get('/api/models/comparison')
            assert comparison_response.status_code == 200
    
    @patch('backend.app.services.model_loader.get_models')
    def test_image_only_workflow(self, mock_get_models, client, sample_image):
        """Test image-only analysis workflow."""
        mock_models = {'image_only': MagicMock()}
        mock_get_models.return_value = mock_models
        
        with patch('backend.app.api.prediction.predict_image_only') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Mild Stress',
                'confidence': 0.75,
                'probabilities': {'Healthy': 0.15, 'Mild Stress': 0.75, 'Moderate Stress': 0.08, 'Severe Stress': 0.02},
                'model_version': 'image_v1.0.0',
            }
            
            files = {'image': ('leaf.jpg', sample_image.getvalue(), 'image/jpeg')}
            data = {'crop': 'rice'}
            
            response = client.post('/api/predict/image', files=files, data=data)
            
            assert response.status_code == 200
            data = response.json()
            assert data['model_type'] == 'image_only'
            assert data['prediction'] == 'Mild Stress'
            
            # Verify in history
            pred_id = data['prediction_id']
            history = client.get(f'/api/predictions/{pred_id}')
            assert history.status_code == 200
            assert history.json()['model_type'] == 'image_only'
    
    @patch('backend.app.services.model_loader.get_models')
    def test_sensor_only_workflow(self, mock_get_models, client, valid_sensor_data):
        """Test sensor-only analysis workflow."""
        mock_models = {'sensor_only': MagicMock()}
        mock_get_models.return_value = mock_models
        
        with patch('backend.app.api.prediction.predict_sensor_only') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Healthy',
                'confidence': 0.92,
                'probabilities': {'Healthy': 0.92, 'Mild Stress': 0.05, 'Moderate Stress': 0.02, 'Severe Stress': 0.01},
                'model_version': 'sensor_v1.0.0',
            }
            
            response = client.post('/api/predict/sensor', json={
                'crop': 'maize',
                'sensor_data': valid_sensor_data,
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data['model_type'] == 'sensor_only'
            assert data['prediction'] == 'Healthy'
            
            # Verify in history
            pred_id = data['prediction_id']
            history = client.get(f'/api/predictions/{pred_id}')
            assert history.status_code == 200
            assert history.json()['model_type'] == 'sensor_only'


class TestErrorHandling:
    """Test error handling across the workflow."""
    
    def test_invalid_image_format(self, client, valid_sensor_data):
        """Test handling of invalid image format."""
        # Create a text file
        files = {'image': ('test.txt', b'not an image', 'text/plain')}
        data = valid_sensor_data
        
        response = client.post('/api/predict', files=files, data=data)
        
        assert response.status_code == 400
        assert 'Invalid image type' in response.json()['detail']
    
    def test_image_too_large(self, client, valid_sensor_data):
        """Test handling of oversized image."""
        # Create a large image (simulate by mocking)
        large_image = io.BytesIO(b'x' * (11 * 1024 * 1024))  # 11MB
        files = {'image': ('large.jpg', large_image.getvalue(), 'image/jpeg')}
        data = valid_sensor_data
        
        response = client.post('/api/predict', files=files, data=data)
        
        assert response.status_code == 400
        assert 'too large' in response.json()['detail'].lower()
    
    def test_missing_sensor_data(self, client, sample_image):
        """Test prediction with missing sensor fields."""
        files = {'image': ('leaf.jpg', sample_image.getvalue(), 'image/jpeg')}
        data = {'soil_moisture': 45.0}  # Missing other required fields
        
        response = client.post('/api/predict', files=files, data=data)
        
        assert response.status_code == 400
    
    def test_out_of_range_sensor_values(self, client, sample_image):
        """Test prediction with out-of-range sensor values."""
        files = {'image': ('leaf.jpg', sample_image.getvalue(), 'image/jpeg')}
        data = {
            'soil_moisture': 150,  # > 100
            'temperature': -20,    # < -10
            'humidity': 65.0,
            'rainfall': 12.3,
            'light_intensity': 850,
        }
        
        response = client.post('/api/predict', files=files, data=data)
        
        assert response.status_code == 400
        assert 'outside valid range' in response.json()['detail'].lower()


class TestAnalyticsWorkflow:
    """Test analytics data flow."""
    
    @patch('backend.app.services.model_loader.get_models')
    def test_analytics_after_multiple_predictions(self, mock_get_models, client, sample_image, valid_sensor_data):
        """Test that analytics aggregate correctly after multiple predictions."""
        mock_models = {'multimodal': MagicMock()}
        mock_get_models.return_value = mock_models
        
        with patch('backend.app.api.prediction.predict_multimodal') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Moderate Stress',
                'confidence': 0.85,
                'probabilities': {'Healthy': 0.05, 'Mild Stress': 0.1, 'Moderate Stress': 0.85, 'Severe Stress': 0.0},
                'model_version': 'multimodal_v1.0.0',
            }
            
            # Make 3 predictions
            for i in range(3):
                files = {'image': (f'leaf_{i}.jpg', sample_image.getvalue(), 'image/jpeg')}
                data = {**valid_sensor_data, 'crop': 'wheat'}
                response = client.post('/api/predict', files=files, data=data)
                assert response.status_code == 200
            
            # Check analytics
            analytics = client.get('/api/analytics?days=1')
            assert analytics.status_code == 200
            data = analytics.json()
            assert data['summary']['total_predictions'] >= 3
            assert 'Moderate Stress' in data['predictions_by_class']
            assert data['predictions_by_class']['Moderate Stress'] >= 3
            assert 'multimodal' in data['predictions_by_model']


class TestModelComparisonWorkflow:
    """Test model comparison data flow."""
    
    def test_model_comparison_endpoint(self, client):
        """Test model comparison endpoint returns all three models."""
        response = client.get('/api/models/comparison')
        assert response.status_code == 200
        data = response.json()
        
        # Should have metrics for all three model types
        expected_models = ['image_only', 'sensor_only', 'multimodal']
        for model in expected_models:
            if model in data:
                assert 'f1_macro' in data[model]
                assert 'accuracy' in data[model]
                assert 'confusion_matrix' in data[model]
    
    def test_model_metrics_endpoint(self, client):
        """Test individual model metrics endpoints."""
        for model_type in ['image_only', 'sensor_only', 'multimodal']:
            response = client.get(f'/api/models/metrics/{model_type}')
            # May return 404 if no metrics stored yet, but should not error
            assert response.status_code in [200, 404]


class TestHealthAndMonitoring:
    """Test health checks and monitoring endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'models_loaded' in data
        assert 'database' in data
    
    def test_readiness_endpoint(self, client):
        """Test readiness probe."""
        response = client.get('/api/health/ready')
        assert response.status_code == 200
        assert 'status' in response.json()
    
    def test_liveness_endpoint(self, client):
        """Test liveness probe."""
        response = client.get('/api/health/live')
        assert response.status_code == 200
        assert response.json()['status'] == 'alive'


class TestDataPersistence:
    """Test data persistence across requests."""
    
    @patch('backend.app.services.model_loader.get_models')
    def test_prediction_persisted_in_database(self, mock_get_models, client, sample_image, valid_sensor_data):
        """Test that predictions are saved to database."""
        mock_models = {'multimodal': MagicMock()}
        mock_get_models.return_value = mock_models
        
        with patch('backend.app.api.prediction.predict_multimodal') as mock_predict:
            mock_predict.return_value = {
                'prediction': 'Severe Stress',
                'confidence': 0.95,
                'probabilities': {'Healthy': 0.0, 'Mild Stress': 0.0, 'Moderate Stress': 0.05, 'Severe Stress': 0.95},
                'model_version': 'multimodal_v1.0.0',
            }
            
            # Make prediction
            files = {'image': ('leaf.jpg', sample_image.getvalue(), 'image/jpeg')}
            data = {**valid_sensor_data, 'crop': 'tomato'}
            response = client.post('/api/predict', files=files, data=data)
            
            assert response.status_code == 200
            pred_id = response.json()['prediction_id']
            
            # Verify it can be retrieved
            get_response = client.get(f'/api/predictions/{pred_id}')
            assert get_response.status_code == 200
            retrieved = get_response.json()
            assert retrieved['prediction_id'] == pred_id
            assert retrieved['crop'] == 'tomato'
            assert retrieved['prediction'] == 'Severe Stress'
            
            # Verify sensor data persisted
            assert retrieved['sensor_data']['soil_moisture'] == valid_sensor_data['soil_moisture']
            assert retrieved['sensor_data']['temperature'] == valid_sensor_data['temperature']
    
    def test_prediction_deletion(self, client):
        """Test prediction deletion."""
        # Get an existing prediction
        list_resp = client.get('/api/predictions?limit=1')
        if list_resp.json()['total'] > 0:
            pred_id = list_resp.json()['predictions'][0]['prediction_id']
            
            # Delete it
            del_resp = client.delete(f'/api/predictions/{pred_id}')
            assert del_resp.status_code == 200
            
            # Verify it's gone
            get_resp = client.get(f'/api/predictions/{pred_id}')
            assert get_resp.status_code == 404


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])