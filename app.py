from fastapi import FastAPI, UploadFile, File
from arcee import arcee
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pypdf
import docx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ARCEE_API_KEY = os.getenv("ARCEE_API_KEY")

# ❗ REMOVE GLOBAL MODEL (IMPORTANT)


# Lazy load embedding model (fix for memory issue)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Only initialize Arcee (lightweight)
try:
    test_response = arcee(
        model="trinity-large-thinking",
        prompt="hi",
        api_key=ARCEE_API_KEY,
        max_tokens=1
    )
    print("Arcee AI connection successful")
    ARCEE_CLIENT = True
except Exception as e:
    print(f"Arcee AI Error: {e}")
    ARCEE_CLIENT = False


# Storage
DOCUMENTS = []
DOC_EMBEDDINGS = None
INDEX = None


# Extract PDF
def extract_pdf(file):
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


# Extract DOCX
def extract_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


# Upload
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global DOCUMENTS, DOC_EMBEDDINGS, INDEX

    if file.filename.endswith(".pdf"):
        text = extract_pdf(file.file)
    elif file.filename.endswith(".docx"):
        text = extract_docx(file.file)
    else:
        return {"error": "Unsupported file type"}

    if not text.strip():
        return {"error": "Empty document"}

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    DOCUMENTS.extend(chunks)

    # 🔥 Lazy load here
    model = get_embedding_model()
    embeddings = model.encode(chunks)

    if DOC_EMBEDDINGS is None:
        DOC_EMBEDDINGS = embeddings
    else:
        DOC_EMBEDDINGS = np.vstack((DOC_EMBEDDINGS, embeddings))

    dimension = embeddings.shape[1]
    INDEX = faiss.IndexFlatL2(dimension)
    INDEX.add(DOC_EMBEDDINGS)

    return {"message": "File processed", "chunks": len(chunks)}


# Query
@app.post("/query")
async def ask(query: str):
    global DOCUMENTS, INDEX

    if INDEX is None:
        return {"error": "No documents uploaded"}

    model = get_embedding_model()  # 🔥 Lazy load again
    query_embedding = model.encode([query])

    distances, indices = INDEX.search(query_embedding, 3)

    context = "\n".join([DOCUMENTS[i] for i in indices[0] if i < len(DOCUMENTS)])

    if not context.strip():
        return {"error": "No relevant context found"}

    if not ARCEE_CLIENT:
        return {
            "answer": f"Demo mode:\n\n{context[:500]}...",
            "context_used": context,
            "demo_mode": True
        }

    try:
        response = arcee(
            model="trinity-large-thinking",
            prompt=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer only from the given context:",
            api_key=ARCEE_API_KEY,
            max_tokens=1000
        )
    except Exception as e:
        return {"error": str(e), "context_used": context}

    return {
        "answer": response,
        "context_used": context
    }