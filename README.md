# 🤖 RAG API with Arcee AI

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Arcee AI](https://img.shields.io/badge/Arcee%20AI-Trinity%20Large-purple.svg)](https://arcee.ai)

## 📝 Description

Advanced **Retrieval-Augmented Generation (RAG)** API that processes documents and answers questions using **Arcee AI's Trinity Large Thinking** model. Features document upload, vector search, and intelligent query responses.

## 🌐 Live Demo

**🚀 API Documentation**: [https://your-rag-api.onrender.com/docs](https://your-rag-api.onrender.com/docs)

*Note: Deploy the app to get your live link*

## ✨ Features

- 📄 **Document Processing**: PDF & DOCX file support
- 🔍 **Vector Search**: FAISS-based similarity search
- 🧠 **Smart AI**: Arcee AI Trinity Large Thinking model
- 🚀 **Fast API**: FastAPI with async support
- 📊 **Chunking**: Intelligent document chunking (500 chars)
- 🔐 **Secure**: API key authentication
- 🌍 **Deployable**: Ready for cloud deployment

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.12+
- **AI Model**: Arcee AI - Trinity Large Thinking
- **Vector DB**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Document Processing**: PyPDF, python-docx
- **Deployment**: Docker, Render

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Shadow-linux-png/ML_API.git
cd ML_API
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create `.env` file:
```env
ARCEE_API_KEY=your_arcee_api_key_here
```

### 4. Run the API
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the API
- **API Docs**: http://localhost:8000/docs
- **Upload Endpoint**: POST `/upload`
- **Query Endpoint**: POST `/query`

## 📖 API Usage

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### Query Document
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is this document about?\"}"
```

## 🏗️ Project Structure

```
ml-api/
├── app.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── .env              # Environment variables
├── dockerfile        # Docker configuration
├── render.yaml       # Render deployment config
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

## 🔧 Configuration

### Environment Variables
- `ARCEE_API_KEY`: Your Arcee AI API key (required)

### Supported File Types
- PDF files (.pdf)
- Word documents (.docx)

### API Endpoints
- `POST /upload` - Upload and process documents
- `POST /query` - Query uploaded documents
- `GET /docs` - Interactive API documentation

## 🚀 Deployment

### Render (Recommended)
1. Fork this repository
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Connect your GitHub repository
4. Set `ARCEE_API_KEY` environment variable
5. Deploy using `render.yaml` configuration

### Docker
```bash
docker build -t rag-api .
docker run -p 8000:8000 rag-api
```

### Manual Deployment
```bash
git clone https://github.com/Shadow-linux-png/ML_API.git
cd ML_API
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 10000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Arcee AI](https://arcee.ai) - For the amazing Trinity Large Thinking model
- [FastAPI](https://fastapi.tiangolo.com) - For the modern web framework
- [Sentence Transformers](https://sbert.net/) - For powerful embeddings
- [FAISS](https://faiss.ai/) - For efficient vector search

## 📞 Support

For any queries or support:
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Shadow-linux-png/ML_API/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Shadow-linux-png/ML_API/discussions)

---

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Shadow-linux-png](https://github.com/Shadow-linux-png)
