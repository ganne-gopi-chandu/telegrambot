# ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "convexopti.pdf")
STORE_DIR = os.path.join(BASE_DIR, "data", "faiss_store")
os.makedirs(STORE_DIR, exist_ok=True)

# 1. Load PDF
docs = PyPDFLoader(PDF_PATH).load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

documents = [{"content": d.page_content, "metadata": d.metadata} for d in chunks]
texts = [d["content"] for d in documents]

# 3. Embed
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(texts, show_progress_bar=True)

# 4. Save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
faiss.write_index(index, f"{STORE_DIR}/index.faiss")

# 5. Save documents
with open(f"{STORE_DIR}/docs.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Ingestion done. Index + docs saved.")

