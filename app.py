from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pickle
import numpy as np
import os
import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS
CORS(app, origins=["http://localhost:3000", "http://localhost:8080"])

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per hour"]
)

# Global variables for model components
model = None
vectorizer = None
model_metadata = {
    'version': '1.0.0',
    'model_type': 'MultinomialNB',
    'feature_extraction': 'TF-IDF',
    'trained_at': None,
    'accuracy': None,
    'classes': ['Negative', 'Positive']
}

# Model configuration
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')

def load_model() -> bool:
    """Load the trained model and vectorizer with error handling."""
    global model, vectorizer, model_metadata
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load metadata if exists
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                model_metadata.update(json.load(f))
        
        logger.info("Model and vectorizer loaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def validate_text_input(text: str) -> Optional[str]:
    """Validate input text for prediction."""
    if not text or not text.strip():
        return "Text cannot be empty"
    
    if len(text.strip()) > 10000:
        return "Text is too long (max 10000 characters)"
    
    if len(text.strip()) < 3:
        return "Text is too short (min 3 characters)"
    
    return None

# Load model on startup
load_model()

@app.route('/predict', methods=['POST', 'GET'])
@limiter.limit("10 per minute")
def predict():
    """Predict sentiment for input text with comprehensive error handling."""
    
    # Handle GET requests for browser testing
    if request.method == 'GET':
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis API</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .form { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                textarea { width: 100%; height: 100px; padding: 10px; margin: 10px 0; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                .result { background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis API</h1>
            <div class="form">
                <h3>Test API</h3>
                <form method="POST" action="/predict">
                    <textarea name="text" placeholder="Enter text to analyze..."></textarea><br>
                    <button type="submit">Analyze Sentiment</button>
                </form>
                <p><strong>Note:</strong> For programmatic access, use POST requests with JSON data.</p>
                <h3>API Endpoints:</h3>
                <ul>
                    <li><strong>POST /predict</strong> - Single text analysis</li>
                    <li><strong>POST /predict/batch</strong> - Batch text analysis</li>
                    <li><strong>GET /health</strong> - Health check</li>
                </ul>
            </div>
        </body>
        </html>
        '''
    
    if model is None or vectorizer is None:
        logger.warning("Prediction attempted with no model loaded")
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'code': 'MODEL_NOT_LOADED'
        }), 503
    
    try:
        # Validate request format
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request',
                'code': 'MISSING_FIELD'
            }), 400
        
        text = data['text']
        
        # Validate text input
        validation_error = validate_text_input(text)
        if validation_error:
            return jsonify({
                'error': validation_error,
                'code': 'INVALID_INPUT'
            }), 400
        
        # Vectorize the input text
        text_vector = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = probabilities.max()
        
        # Get class probabilities
        class_probs = {
            model_metadata['classes'][i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Log prediction
        logger.info(f"Prediction made: {prediction} with confidence {confidence:.4f}")
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': model_metadata['classes'][prediction],
            'confidence': float(confidence),
            'probabilities': class_probs,
            'text': text,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': model_metadata['version']
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error during prediction',
            'code': 'PREDICTION_ERROR',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Comprehensive health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'model_metadata': model_metadata,
        'api_version': '2.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/predict/batch', methods=['POST'])
@limiter.limit("5 per minute")
def predict_batch():
    """Batch prediction for multiple texts."""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'code': 'MODEL_NOT_LOADED'
        }), 503
    
    try:
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({
                'error': 'Missing "texts" field in request',
                'code': 'MISSING_FIELD'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': '"texts" must be an array',
                'code': 'INVALID_INPUT'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'error': 'Too many texts (max 100 per batch)',
                'code': 'BATCH_TOO_LARGE'
            }), 400
        
        # Validate all texts
        for i, text in enumerate(texts):
            validation_error = validate_text_input(text)
            if validation_error:
                return jsonify({
                    'error': f'Invalid text at index {i}: {validation_error}',
                    'code': 'INVALID_INPUT'
                }), 400
        
        # Batch prediction
        text_vectors = vectorizer.transform(texts)
        predictions = model.predict(text_vectors)
        probabilities = model.predict_proba(text_vectors)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            class_probs = {
                model_metadata['classes'][j]: float(prob) 
                for j, prob in enumerate(probs)
            }
            
            results.append({
                'index': i,
                'text': texts[i],
                'prediction': int(pred),
                'prediction_label': model_metadata['classes'][pred],
                'confidence': float(probs.max()),
                'probabilities': class_probs
            })
        
        logger.info(f"Batch prediction completed for {len(texts)} texts")
        
        return jsonify({
            'results': results,
            'batch_size': len(texts),
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': model_metadata['version']
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error during batch prediction',
            'code': 'BATCH_PREDICTION_ERROR',
            'details': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get detailed model information."""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'code': 'MODEL_NOT_LOADED'
        }), 503
    
    info = {
        'metadata': model_metadata,
        'model_type': type(model).__name__,
        'vectorizer_type': type(vectorizer).__name__,
        'feature_count': vectorizer.get_feature_names_out().shape[0],
        'classes': model_metadata['classes'],
        'model_path': MODEL_PATH,
        'vectorizer_path': VECTORIZER_PATH
    }
    
    return jsonify(info)

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the model from disk."""
    if load_model():
        logger.info("Model reloaded successfully")
        return jsonify({
            'message': 'Model reloaded successfully',
            'timestamp': datetime.utcnow().isoformat()
        })
    else:
        logger.error("Failed to reload model")
        return jsonify({
            'error': 'Failed to reload model',
            'code': 'RELOAD_FAILED'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'code': 'NOT_FOUND',
        'available_endpoints': [
            'GET /health',
            'POST /predict',
            'POST /predict/batch',
            'GET /model/info',
            'POST /model/reload'
        ]
    }), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'code': 'RATE_LIMIT_EXCEEDED',
        'message': str(e.description)
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (for production)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting ML API server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Model loaded: {model is not None}")
    
    if args.debug:
        app.run(debug=True, host=args.host, port=args.port)
    else:
        from waitress import serve
        logger.info("Running in production mode with waitress")
        serve(app, host=args.host, port=args.port, threads=args.workers)
