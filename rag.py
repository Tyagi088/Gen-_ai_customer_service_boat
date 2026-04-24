import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBED_MODEL, FAISS_PATH

model = SentenceTransformer(EMBED_MODEL)

index = faiss.read_index(FAISS_PATH)

documents = []  # You can persist this separately

def retrieve(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=5)
    return [documents[i] for i in I[0]]

def generate(query, context):
    return f"Answer based on context: {context[:500]}"
