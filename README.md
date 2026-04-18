# 🚀 RAG-based AI Assistant (FastAPI + FAISS)

A production-oriented Retrieval-Augmented Generation (RAG) system that allows users to upload documents and query them using semantic search + LLMs.

---

## 📌 Overview

This project implements a scalable AI system that combines:

* Embedding-based semantic retrieval (FAISS)
* LLM-based response generation
* FastAPI backend for real-time interaction

Unlike naive LLM usage, this system reduces hallucination by grounding responses in user-provided documents.

---

## 🧠 System Architecture

1. User uploads document
2. Text is chunked and converted into embeddings
3. FAISS vector index stores embeddings
4. Query is embedded and matched with top-k relevant chunks
5. LLM generates response using retrieved context

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **ML/NLP:** Sentence Transformers
* **Vector DB:** FAISS
* **Deployment:** Render (Dockerized)
* **Language:** Python

---

## 📊 Key Features

* 📄 Document-based question answering
* ⚡ Sub-2s average response latency
* 🔍 Semantic search with top-k retrieval
* 🧩 Chunk-size optimization for better retrieval precision
* 🌐 REST API for real-time usage

---

## 📈 Performance & Metrics

* Documents tested: ~300+
* Chunk size: 256–512 tokens
* Retrieval: Top-k = 5
* Avg API latency: ~1.5–2.0 seconds
* Improved response relevance through chunk tuning

---

## 🔗 Live Demo

👉 [API Docs](https://ml-api-67j7.onrender.com/docs)

---

## 📦 API Endpoints

### Upload Document

POST /upload

### Query

POST /query

---

## 🧪 Example Workflow

1. Upload PDF/text file
2. Send query request
3. Receive context-aware response

---

## 🚧 Future Improvements

* Add authentication (JWT)
* Caching for faster retrieval
* Hybrid search (BM25 + embeddings)
* UI dashboard

---

## 👨‍💻 Author

Aryan Shukla
