# ML API - Production Ready

A comprehensive, production-ready machine learning API built with Flask for text classification using sentiment analysis. This API includes enterprise-grade features such as rate limiting, CORS support, comprehensive logging, batch processing, and containerization.

## 🚀 Features

- **Production Ready**: Built with enterprise best practices
- **Multiple Endpoints**: Single and batch prediction capabilities
- **Rate Limiting**: Built-in rate limiting with Redis support
- **CORS Support**: Cross-origin resource sharing enabled
- **Comprehensive Logging**: Structured logging with file and console output
- **Error Handling**: Detailed error responses with error codes
- **Health Checks**: Multiple health check endpoints
- **Model Management**: Model reloading and metadata tracking
- **Containerized**: Docker and Docker Compose support
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: Detailed API documentation

## 📁 Project Structure

```
ml-api/
├── app.py                 # Main Flask API server
├── train.py              # Model training script with hyperparameter tuning
├── test_api.py           # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── Dockerfile           # Docker container configuration
├── docker-compose.yml   # Multi-container orchestration
├── nginx.conf           # Nginx reverse proxy configuration
├── README.md            # This file
├── models/              # Trained model files (auto-created)
│   ├── model.pkl        # Trained ML model
│   ├── vectorizer.pkl   # Text vectorizer
│   └── metadata.json    # Model metadata
└── logs/                # Application logs (auto-created)
    └── api.log          # API access logs
```

## 🛠️ Setup Instructions

### 1. Clone and Setup

```bash
cd ml-api
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Train the Model

```bash
python train.py
```

This will:
- Train a Naive Bayes classifier with hyperparameter tuning
- Perform cross-validation and comprehensive evaluation
- Create and save model files in the `models/` directory
- Generate detailed training logs and metadata

### 5. Run the API Server

#### Development Mode
```bash
python app.py --debug
```

#### Production Mode
```bash
python app.py --workers 4
```

#### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build and run individual containers
docker build -t ml-api .
docker run -p 5000:5000 ml-api
```

## 📡 API Endpoints

### Core Endpoints

#### POST `/predict`
Predict sentiment for a single text.

**Request:**
```json
{
    "text": "I love this product!"
}
```

**Response:**
```json
{
    "prediction": 1,
    "prediction_label": "Positive",
    "confidence": 0.95,
    "probabilities": {
        "Negative": 0.02,
        "Positive": 0.95,
        "Neutral": 0.03
    },
    "text": "I love this product!",
    "timestamp": "2023-12-01T10:00:00.000Z",
    "model_version": "2.0.0"
}
```

#### POST `/predict/batch`
Predict sentiment for multiple texts (max 100 per batch).

**Request:**
```json
{
    "texts": [
        "I love this product!",
        "This is terrible",
        "It's okay"
    ]
}
```

**Response:**
```json
{
    "results": [
        {
            "index": 0,
            "text": "I love this product!",
            "prediction": 1,
            "prediction_label": "Positive",
            "confidence": 0.95,
            "probabilities": {"Negative": 0.02, "Positive": 0.95, "Neutral": 0.03}
        },
        ...
    ],
    "batch_size": 3,
    "timestamp": "2023-12-01T10:00:00.000Z",
    "model_version": "2.0.0"
}
```

### Management Endpoints

#### GET `/health`
Comprehensive health check.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "vectorizer_loaded": true,
    "model_metadata": {
        "version": "2.0.0",
        "model_type": "MultinomialNB",
        "feature_extraction": "TF-IDF",
        "trained_at": "2023-12-01T09:30:00.000Z",
        "test_accuracy": 0.95,
        "classes": ["Negative", "Positive", "Neutral"]
    },
    "api_version": "2.0.0",
    "timestamp": "2023-12-01T10:00:00.000Z"
}
```

#### GET `/model/info`
Detailed model information.

#### POST `/model/reload`
Reload the model from disk.

## 🧪 Testing

### Run All Tests
```bash
pytest test_api.py -v
```

### Run Specific Test Classes
```bash
pytest test_api.py::TestPredictionEndpoint -v
```

### Run with Coverage
```bash
pytest --cov=app test_api.py -v
```

## 📊 Model Details

- **Algorithm**: Multinomial Naive Bayes with hyperparameter tuning
- **Feature Extraction**: TF-IDF Vectorization with n-grams
- **Task**: Multi-class sentiment classification (Negative/Positive/Neutral)
- **Training**: Grid search with cross-validation
- **Evaluation**: Comprehensive metrics including confusion matrix

### Model Performance

The training script provides:
- **Cross-validation**: 5-fold CV with accuracy metrics
- **Hyperparameter tuning**: Grid search for optimal parameters
- **Test set evaluation**: Accuracy, precision, recall, F1-score
- **Confusion matrix**: Detailed classification analysis

## 🔧 Configuration

### Environment Variables

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_WORKERS=4

# Rate Limiting
RATELIMIT_STORAGE_URL=redis://localhost:6379
RATELIMIT_DEFAULT=100 per hour

# Model Configuration
MODEL_DIR=models
MODEL_RELOAD_ON_STARTUP=False

# Logging
LOG_LEVEL=INFO
LOG_FILE=api.log

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Rate Limiting

- **Default**: 100 requests per hour per IP
- **Prediction**: 10 requests per minute per IP
- **Batch Prediction**: 5 requests per minute per IP

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Docker Commands

```bash
# Build image
docker build -t ml-api .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models ml-api

# Run with environment variables
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e API_WORKERS=4 \
  -v $(pwd)/models:/app/models \
  ml-api
```

## 📈 Production Deployment

### Using Gunicorn

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Using Waitress (Windows)

```bash
waitress-serve --host=0.0.0.0 --port=5000 --threads=4 app:app
```

### Nginx Reverse Proxy

The included `nginx.conf` provides:
- Load balancing
- Rate limiting at the web server level
- Security headers
- Request logging
- Health check bypass

## 🔍 Monitoring and Logging

### Application Logs

- **Location**: `logs/api.log`
- **Format**: Structured JSON logging
- **Levels**: INFO, WARNING, ERROR
- **Rotation**: Configure with logrotate in production

### Health Monitoring

- **Endpoint**: `/health`
- **Metrics**: Model status, API version, timestamp
- **Monitoring**: Integrate with Prometheus or similar tools

### Error Tracking

- **Error Codes**: Standardized error responses
- **Logging**: Full stack traces for debugging
- **Monitoring**: Alert on 5xx errors

## 🔒 Security Features

- **Input Validation**: Comprehensive text validation
- **Rate Limiting**: Prevent abuse and DoS attacks
- **CORS**: Configurable cross-origin policies
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, XSS protection
- **Input Sanitization**: Length limits and content validation

## 🚀 Performance Optimization

- **Batch Processing**: Efficient handling of multiple predictions
- **Caching**: Model loading optimization
- **Connection Pooling**: Database connection management
- **Async Processing**: Non-blocking request handling
- **Memory Management**: Efficient model loading and unloading

## 📝 API Usage Examples

### Python Client

```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'Great product!'})
result = response.json()
print(f"Prediction: {result['prediction_label']} ({result['confidence']:.2f})")

# Batch prediction
response = requests.post('http://localhost:5000/predict/batch',
                        json={'texts': ['Great!', 'Terrible!', 'Okay']})
results = response.json()['results']
for result in results:
    print(f"{result['text']}: {result['prediction_label']} ({result['confidence']:.2f})")

# Health check
response = requests.get('http://localhost:5000/health')
print(f"API Status: {response.json()['status']}")
```

### JavaScript Client

```javascript
// Single prediction
const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'Great product!' })
});
const result = await response.json();
console.log(`Prediction: ${result.prediction_label} (${result.confidence})`);

// Batch prediction
const batchResponse = await fetch('http://localhost:5000/predict/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        texts: ['Great!', 'Terrible!', 'Okay'] 
    })
});
const batchResults = await batchResponse.json();
batchResults.results.forEach(result => {
    console.log(`${result.text}: ${result.prediction_label} (${result.confidence})`);
});
```

### cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!", "Okay"]}'

# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model/info

# Reload model
curl -X POST http://localhost:5000/model/reload
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model not found**: Run `python train.py` to train and save the model
2. **Port already in use**: Change the port with `--port` argument
3. **Redis connection failed**: Ensure Redis is running for rate limiting
4. **Memory issues**: Reduce batch size or model complexity

### Debug Mode

```bash
python app.py --debug --host 127.0.0.1 --port 5000
```

### Log Analysis

```bash
# View real-time logs
tail -f logs/api.log

# Filter error messages
grep "ERROR" logs/api.log

# Analyze prediction patterns
grep "Prediction made" logs/api.log | wc -l
```
