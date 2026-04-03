from fastapi import FastAPI, UploadFile, File
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pypdf
import docx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=GROQ_API_KEY)

# Storage
DOCUMENTS = []
DOC_EMBEDDINGS = None
INDEX = None


# Extract PDF
def extract_pdf(file):
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
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

    # Chunk
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    DOCUMENTS.extend(chunks)

    # Embeddings
    embeddings = embedding_model.encode(chunks)

    if DOC_EMBEDDINGS is None:
        DOC_EMBEDDINGS = embeddings
    else:
        DOC_EMBEDDINGS = np.vstack((DOC_EMBEDDINGS, embeddings))

    # FAISS
    dimension = embeddings.shape[1]
    INDEX = faiss.IndexFlatL2(dimension)
    INDEX.add(DOC_EMBEDDINGS)

    return {"message": "File processed", "chunks": len(chunks)}


# Query
@app.post("/query")
def ask(query: str):
    global DOCUMENTS, INDEX

    if INDEX is None:
        return {"error": "No documents uploaded"}

    query_embedding = embedding_model.encode([query])
    distances, indices = INDEX.search(query_embedding, 3)

    context = "\n".join([DOCUMENTS[i] for i in indices[0]])

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Answer only from given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{query}"}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "context_used": context
    }