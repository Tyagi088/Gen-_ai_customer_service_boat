import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_PATH = "data/sources.json"
FAISS_PATH = "vectorstore/faiss_index"
