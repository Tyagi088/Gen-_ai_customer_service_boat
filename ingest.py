import json
import requests
import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.utils import hash_text
from app.config import DATA_PATH, EMBED_MODEL, FAISS_PATH

model = SentenceTransformer(EMBED_MODEL)

def load_sources():
    with open(DATA_PATH) as f:
        return json.load(f)

def fetch_data(url):
    res = requests.get(url)
    return res.text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def load_index():
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    return faiss.IndexFlatL2(384)

def save_index(index):
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, FAISS_PATH)

def run_ingestion():
    sources = load_sources()
    index = load_index()

    existing_hashes = set()

    for src in sources:
        text = fetch_data(src["url"])
        h = hash_text(text)

        if h in existing_hashes:
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks)

        index.add(np.array(embeddings))

        existing_hashes.add(h)

    save_index(index)
    print("✅ Knowledge base updated")
