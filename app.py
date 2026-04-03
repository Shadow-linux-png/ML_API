from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from datetime import datetime
import numpy as np

app = FastAPI()

# Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Load model
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
except:
    model = None
    vectorizer = None

# Request schema
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

# ------------------- ROUTES -------------------

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict")
def predict(data: TextInput):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = data.text.strip()

    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Text too short")

    if len(text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long")

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    return {
        "input": text,
        "prediction": int(pred),
        "confidence": float(max(probs)),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(data.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts allowed")

    vecs = vectorizer.transform(data.texts)
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)

    results = []

    for i, (p, pr) in enumerate(zip(preds, probs)):
        results.append({
            "index": i,
            "text": data.texts[i],
            "prediction": int(p),
            "confidence": float(max(pr))
        })

    return {
        "results": results,
        "batch_size": len(results)
    }
