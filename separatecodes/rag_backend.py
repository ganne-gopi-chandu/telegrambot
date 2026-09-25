# rag_backend.py
import os
import json
import sqlite3
import numpy as np
import faiss
import wikipedia
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
import sqlite3
import time
import pickle


load_dotenv()

# ---------------- EMBEDDING ----------------

class EmbeddingManager:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False)

embedding_manager = EmbeddingManager()

# ---------------- VECTOR STORE ----------------

# this is sql+vector store
STORE_DIR = "./data/faiss_store"


class VectorStore:
    """
    Large-scale FAISS Vector Store
    - Cosine similarity
    - IVF + HNSW (fast & scalable)
    - SQLite metadata storage
    """

    def __init__(
        self,
        persist_directory: str = "./data/faiss_store",
        embedding_dim: int = 768,
        nlist: int = 4096,          # number of clusters
        hnsw_m: int = 32,
        embeddings: np.ndarray = None
    ):
        self.persist_directory = persist_directory
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.hnsw_m = hnsw_m
        
        index_file = os.path.join(self.persist_directory, "index.faiss")
        docs_file = os.path.join(self.persist_directory, "docs.pkl")
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            self.index = None

        if os.path.exists(docs_file):
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
        else:
            self.documents = []


        os.makedirs(self.persist_directory, exist_ok=True)

        self.index_path = os.path.join(self.persist_directory, "index.faiss")
        self.db_path = os.path.join(self.persist_directory, "metadata.db")
        self.is_trained = False
        self._init_db()
        if os.path.exists(self.index_path):
            self._init_faiss()  # load existing index
        elif embeddings is not None:
            self._init_faiss(embeddings)  # create new index
        else:
            raise ValueError("Embeddings required to create a new FAISS index")


    # -------------------- DATABASE --------------------

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    # -------------------- FAISS --------------------

    def _init_faiss(self, embeddings: np.ndarray = None):
        if os.path.exists(self.index_path):
            loaded_index = faiss.read_index(self.index_path)
            if isinstance(loaded_index, faiss.IndexIDMap2):
                self.index = loaded_index
            else:
                if loaded_index.ntotal == 0:
                    self.index = faiss.IndexIDMap2(loaded_index)  # only empty index can be wrapped
                else:
                    self.index = loaded_index  # already has vectors, cannot wrap
    
            print("FAISS index loaded")
            self.is_trained = self.index.is_trained
            return

        if embeddings is None:
            raise ValueError("Embeddings required to create a new FAISS index")

        # Normalize
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # Adjust nlist
        num_vectors = embeddings.shape[0]
        self.nlist = min(self.nlist, max(1, int(np.sqrt(num_vectors))))

        # Choose index type
        if num_vectors < 1000:
            flat_index = faiss.IndexFlatIP(self.embedding_dim)  # inner product
            self.index = faiss.IndexIDMap2(flat_index)          # wrap with IDMap
            self.is_trained = True                               # flat index does not require training

        else:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            ivf_index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
                )
            self.index = faiss.IndexIDMap2(ivf_index)
            self.index.train(embeddings)   # must train before add_with_ids
            self.is_trained = True 

        # Add embeddings
        ids = np.arange(num_vectors, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)
        self.is_trained = True
        self._persist_index()
    # -------------------- ADD --------------------
    def train(self, embeddings: np.ndarray):
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        if isinstance(self.index, faiss.IndexIDMap2):
            inner_index = self.index.index
            if isinstance(inner_index, faiss.IndexIVFFlat):
                if embeddings.shape[0] < self.nlist:
                    raise ValueError(f"Need at least {self.nlist} vectors to train IVF index")
                inner_index.train(embeddings)
                self.is_trained = True
                self._persist_index()
                print("FAISS index trained successfully")
            else:
                print("Flat index does not require training")
                self.is_trained = True
        else:
            raise RuntimeError("Index must be IndexIDMap2 to train embeddings")



    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray
    ):
        assert len(documents) == len(embeddings), "Size mismatch"
        ids=[]
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # Train if needed
        if not self.index.is_trained:
            raise RuntimeError("FAISS index not trained. Call vectorstore.train() first.")
        
        for doc in documents:
            if isinstance(doc, dict):
                content = doc["content"]
                # metadata = str(doc.get("metadata", {}))
                metadata = json.dumps(doc.get("metadata", {}))

            else:
                content = doc.page_content
                metadata = str(doc.metadata)
            self.cursor.execute("INSERT INTO documents (content, metadata) VALUES (?, ?)",(content, metadata))

            ids.append(self.cursor.lastrowid)

        self.conn.commit()

        # Make sure index is IDMap
        if not isinstance(self.index, faiss.IndexIDMap2):
            raise RuntimeError("FAISS index must be an IndexIDMap2 to use add_with_ids")

        # Prepare embeddings and ids
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        ids = np.array(ids, dtype=np.int64)

        # Add to FAISS
        self.index.add_with_ids(embeddings, ids)


        self._persist_index()
        print(f"Added {len(ids)} documents")

    # -------------------- SEARCH --------------------

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        nprobe: int = 10
    ):
        query_embedding = query_embedding.astype("float32").reshape(1,-1)
        faiss.normalize_L2(query_embedding)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe


        scores, ids = self.index.search(query_embedding, top_k)

        results = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id == -1:
                continue

            self.cursor.execute(
                "SELECT content, metadata FROM documents WHERE id = ?",
                (int(doc_id),)
            )
            row = self.cursor.fetchone()

            if row:
                results.append({
                    "id": int(doc_id),
                    "score": float(score),
                    "content": row[0],
                    "metadata": json.loads(row[1]) if row[1] else {}  #row[1]
                })

        return results

    # -------------------- DELETE --------------------

    def delete(self, doc_ids: List[int]):
        remove_ids = np.array(doc_ids, dtype=np.int64)
        self.index.remove_ids(remove_ids)

        self.cursor.executemany(
            "DELETE FROM documents WHERE id = ?",
            [(int(i),) for i in doc_ids]
        )
        self.conn.commit()

        self._persist_index()
        print(f"Deleted documents: {doc_ids}")

    # -------------------- SAVE --------------------

    def _persist_index(self):
        faiss.write_index(self.index, self.index_path)
    def close(self):
        self.conn.close()

# ---------------- RETRIEVER ----------------

class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5, score_threshold=0.3, nprobe=10):
        q_emb = self.embedding_manager.generate_embedding([query])[0]
        q_emb = np.array([q_emb], dtype="float32")

        results = self.vector_store.similarity_search(q_emb, top_k=top_k, nprobe=nprobe)
        return [r for r in results if r["score"] >= score_threshold]


# ---------------- SERP CLIENT ----------------

class SerpClient:
    def __init__(self, api_key, engines=("google", "youtube")):
        self.api_key = api_key
        self.engines = engines

    def web_search(self, query, num=3):
        search = GoogleSearch({"q": query, "api_key": self.api_key, "engine": "google", "num": num})
        return search.get_dict().get("organic_results", [])

    def youtube_search(self, query, num=3):
        search = GoogleSearch({"q": query, "api_key": self.api_key, "engine": "google_videos", "num": num})
        return search.get_dict().get("video_results", [])


# ---------------- ADVANCED RAG ----------------

# -----------------------------
# Utilities
# -----------------------------

class ToolScore:
    def __init__(self):
        self.scores = {
            "rag": 0.0,
            "wiki": 0.0,
            "web": 0.0
        }

    def normalize(self):
        total = sum(self.scores.values()) + 1e-6
        for k in self.scores:
            self.scores[k] /= total
        return self.scores


# -----------------------------
# Advanced RAG Pipeline
# -----------------------------

class AdvancedRagPipeline:
    def __init__(self, retriever, llm, serp_api=None):
        self.retriever = retriever      # FAISS + SQLite retriever
        self.llm = llm                  # Grok client
        self.serp_api = serp_api
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.history = []

    # -----------------------------
    # Query Rewrite
    # -----------------------------

    def rewrite_query(self, question: str) -> str:
        prompt = f"""
Rewrite the user query to maximize document retrieval quality.
Return only the rewritten query.

Original: {question}
"""
        return self.llm.invoke([prompt]).content.strip()

    # -----------------------------
    # LLM Router
    # -----------------------------

    def llm_route(self, question: str) -> str:
        prompt = f"""
Classify the user query into one of:
- internal_rag
- encyclopedic
- web_search
- hybrid

Return only the label.

Query: "{question}"
"""
        return self.llm.invoke([prompt]).content.strip()

    # -----------------------------
    # Tools
    # -----------------------------

    def wiki_search(self, question: str):
        try:
            pages = wikipedia.search(question, results=1)
            if not pages:
                return None, None
            title = pages[0]
            text = wikipedia.summary(title, sentences=5)
            return text, title
        except Exception:
            return None, None

    # def websearchs(self, question: str):
    #     if not self.serp_api:
    #         return []
    #     if self.wants_video(question):
    #         results = self.serp_api.youtube_search(question)
    #     else:
    #         results = self.serp_api.web_search(question)
    #     return [r.get("snippet", "") for r in results]

    VIDEO_KEYWORDS = [
        "video", "youtube", "watch", "tutorial", "song", "songs","music", "movie", "trailer", "clip", "play",
        "on youtube","from youtube","youtube video","watch on youtube","music video","movie song"]


    def wants_video(self, query: str, keywords=VIDEO_KEYWORDS) -> bool:
        q = query.lower()
        return any(k in q for k in keywords)


    # -----------------------------
    # Reranker
    # -----------------------------

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:5]]
    
    def rerank_with_scores(self, query, docs):
        pairs = [(query, d) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked


    # -----------------------------
    # Real Streaming Output (Grok style)
    # -----------------------------

    def stream_answer(self, prompt: str):
        for token in self.llm.stream(prompt):  # Your Grok SDK streaming call
            yield token

    # -----------------------------
    # Main Query
    # -----------------------------
    def compute_tool_confidence(self, rag_docs, wiki_text, web_snippets, rag_scores=None):
        scores = {"rag": 0.0, "wiki": 0.0, "web": 0.0}

        # RAG contribution
        if rag_docs:
            if rag_scores:
                scores["rag"] = float(sum(max(s, 0.0) for s in rag_scores))
            else:
                scores["rag"] = 1.0

        # Wiki contribution
        if wiki_text:
            scores["wiki"] = 1.0

        # Web contribution
        if web_snippets:
            scores["web"] = min(1.0, len(web_snippets) * 0.3)

        total = sum(scores.values())
        if total == 0:
            return {"rag": 0.0, "wiki": 0.0, "web": 0.0}

        return {k: round(v / total, 3) for k, v in scores.items()}




    def query(self, question: str, top_k=5, min_score=0.1, stream=False):

        # 1️⃣ Query rewrite
        rewritten_query = self.rewrite_query(question)

        # 2️⃣ LLM-based router
        route = self.llm_route(rewritten_query)

        #tool_scores = ToolScore()  not required for now ---
        context_chunks = []
        sources = []

        # 3️⃣ FAISS + SQLite retrieval
        rag_docs = []
        rag_scores=[]
        rag_results = self.retriever.retrieve(rewritten_query, top_k=top_k, score_threshold=min_score)

        if rag_results:
            rag_docs = [doc["content"] for doc in rag_results]
            rag_docs = [d for d in rag_docs if len(d.strip()) > 200]
            if rag_docs:
                reranked = self.rerank_with_scores(rewritten_query, rag_docs)
                rag_docs   = [doc for doc, score in reranked]
                rag_scores = [float(score) for doc, score in reranked]
                for i, _ in enumerate(rag_docs):
                    sources.append({"source": "rag", "page": f"chunk_{i+1}"})


        # 4️⃣ Wikipedia
        WIKI_TRIGGER_SCORE=0.3
        wiki_text, wiki_title = None,None
        if route in ["encyclopedic", "hybrid"]:
            rag_top_score = max(rag_scores, default=0.0)
            if not rag_docs or rag_top_score < WIKI_TRIGGER_SCORE:
                wiki_text, wiki_title = self.wiki_search(rewritten_query)
                if wiki_text:
                    context_chunks.append(wiki_text)
                    sources.append({"source": "wikipedia", "page": wiki_title})
        else:
            wiki_text = None  # for tool_confidence

        web_results = []
        web_snippets = []

        if self.serp_api and route in ["web_search", "hybrid"]:

            # Decide source: Google Web or YouTube
            if self.wants_video(rewritten_query):
                web_results = self.serp_api.youtube_search(rewritten_query)
            else:
                web_results = self.serp_api.web_search(rewritten_query)

            web_snippets = [
                f"{r.get('title', '')} — {r.get('snippet', '')}\nURL: {r.get('link')}"
                for r in web_results
                if r.get("snippet") and r.get("link")
                ]

            context_chunks.extend(web_snippets[:3])

            for i, r in enumerate(web_results[:3]):
                sources.append({
                    "source": "web",        # 👈 stays web
                    "page": f"result_{i+1}",
                    "url": r.get("link"),
                    "title": r.get("title")
                    })

        # 6️⃣ Merge + rerank context
        context_chunks.extend(rag_docs)
        context_chunks = self.rerank(question, context_chunks)

        context = "\n\n".join(context_chunks)


        # ✅ 7️⃣ Honest tool confidence (NO tool_scores, NO heuristics)
        tool_confidence = self.compute_tool_confidence(
            rag_docs=rag_docs,
            wiki_text=wiki_text,
            web_snippets=web_snippets,
            rag_scores=rag_scores
            )

        if not context:
            return {
        "answer": "No relevant context found.",
        "route": route,
        "tool_confidence": tool_confidence
    }

        prompt = f"""
        Answer the question using the context below.
        Context:
        {context}
        Question:
        {question}
        """

        # 6️⃣ Streaming or normal response
        if stream:
            return {
                "route": route,
                "stream": self.stream_answer(prompt),
                "tool_confidence": tool_confidence
            }

        response = self.llm.invoke([prompt]).content

        self.history.append({
            "question": question,
            "rewritten_query": rewritten_query,
            "route": route,
            "tool_confidence": tool_confidence
        })

        return {
            "answer": response,
            "route": route,
            "tool_confidence": tool_confidence
        }
print("complete")
