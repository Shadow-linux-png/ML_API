import pytest
import json
import os
import sys

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_data():
    """Sample test data for predictions."""
    return {
        'positive_text': "I love this product, it's amazing!",
        'negative_text': "This is terrible, I hate it",
        'neutral_text': "The product is okay, nothing special",
        'empty_text': "",
        'short_text': "Hi",
        'long_text': "This is a very long text that exceeds the normal length limits for testing purposes and should trigger validation errors when submitted to the API endpoint for text classification and sentiment analysis predictions."
    }

class TestHealthEndpoint:
    """Test cases for the health check endpoint."""
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'api_version' in data
        assert 'timestamp' in data

class TestPredictionEndpoint:
    """Test cases for the prediction endpoint."""
    
    def test_predict_positive(self, client, sample_data):
        """Test prediction with positive text."""
        response = client.post('/predict',
                             data=json.dumps({'text': sample_data['positive_text']}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'text' in data
        assert 'timestamp' in data
        assert data['text'] == sample_data['positive_text']
    
    def test_predict_negative(self, client, sample_data):
        """Test prediction with negative text."""
        response = client.post('/predict',
                             data=json.dumps({'text': sample_data['negative_text']}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'confidence' in data
        assert data['text'] == sample_data['negative_text']
    
    def test_predict_missing_text(self, client):
        """Test prediction with missing text field."""
        response = client.post('/predict',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'code' in data
        assert data['code'] == 'MISSING_FIELD'
    
    def test_predict_empty_text(self, client, sample_data):
        """Test prediction with empty text."""
        response = client.post('/predict',
                             data=json.dumps({'text': sample_data['empty_text']}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'INVALID_INPUT'
    
    def test_predict_short_text(self, client, sample_data):
        """Test prediction with text that's too short."""
        response = client.post('/predict',
                             data=json.dumps({'text': sample_data['short_text']}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'INVALID_INPUT'
    
    def test_predict_long_text(self, client, sample_data):
        """Test prediction with text that's too long."""
        response = client.post('/predict',
                             data=json.dumps({'text': sample_data['long_text']}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'INVALID_INPUT'
    
    def test_predict_invalid_content_type(self, client, sample_data):
        """Test prediction with invalid content type."""
        response = client.post('/predict',
                             data={'text': sample_data['positive_text']})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'INVALID_CONTENT_TYPE'
    
    def test_predict_malformed_json(self, client):
        """Test prediction with malformed JSON."""
        response = client.post('/predict',
                             data='invalid json',
                             content_type='application/json')
        
        assert response.status_code == 400

class TestBatchPredictionEndpoint:
    """Test cases for the batch prediction endpoint."""
    
    def test_batch_predict_valid(self, client, sample_data):
        """Test batch prediction with valid texts."""
        texts = [
            sample_data['positive_text'],
            sample_data['negative_text'],
            sample_data['neutral_text']
        ]
        
        response = client.post('/predict/batch',
                             data=json.dumps({'texts': texts}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        assert 'batch_size' in data
        assert len(data['results']) == 3
        assert data['batch_size'] == 3
        
        for i, result in enumerate(data['results']):
            assert 'prediction' in result
            assert 'confidence' in result
            assert 'text' in result
            assert result['text'] == texts[i]
    
    def test_batch_predict_missing_texts(self, client):
        """Test batch prediction with missing texts field."""
        response = client.post('/predict/batch',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'MISSING_FIELD'
    
    def test_batch_predict_not_array(self, client):
        """Test batch prediction with non-array texts."""
        response = client.post('/predict/batch',
                             data=json.dumps({'texts': 'not an array'}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'INVALID_INPUT'
    
    def test_batch_predict_too_large(self, client):
        """Test batch prediction with too many texts."""
        texts = [f"Text {i}" for i in range(101)]  # 101 texts, max is 100
        
        response = client.post('/predict/batch',
                             data=json.dumps({'texts': texts}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'BATCH_TOO_LARGE'

class TestModelInfoEndpoint:
    """Test cases for the model info endpoint."""
    
    def test_model_info(self, client):
        """Test the model info endpoint."""
        response = client.get('/model/info')
        
        if response.status_code == 503:  # Model not loaded
            data = json.loads(response.data)
            assert 'error' in data
            assert data['code'] == 'MODEL_NOT_LOADED'
        else:
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'metadata' in data
            assert 'model_type' in data
            assert 'vectorizer_type' in data

class TestModelReloadEndpoint:
    """Test cases for the model reload endpoint."""
    
    def test_model_reload(self, client):
        """Test the model reload endpoint."""
        response = client.post('/model/reload')
        
        if response.status_code == 500:  # Reload failed
            data = json.loads(response.data)
            assert 'error' in data
            assert data['code'] == 'RELOAD_FAILED'
        else:
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'message' in data
            assert 'timestamp' in data

class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'code' in data
        assert data['code'] == 'NOT_FOUND'
        assert 'available_endpoints' in data

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
