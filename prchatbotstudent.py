import os
from dotenv import load_dotenv
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
import wikipedia
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import faiss
import sqlite3
import json
import uuid
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from dotenv import load_dotenv
from typing import List, Dict, Any
import time
import asyncio
from serpapi import GoogleSearch


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "")
serp_api_key = os.getenv("SERP_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

dir_loader = DirectoryLoader(
    PDF_DIR if os.path.exists(PDF_DIR) else BASE_DIR,
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
#dir_loader_pdf=PyPDFLoader(os.path.join(BASE_DIR, "convexopti.pdf"))
#pdf_documents = dir_loader.load()

## textsplitting getting into chunks
def split_documents(documents,chunk_size=800,chunk_overlap=150):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )
    split_docs=text_splitter.split_documents(documents)
    return split_docs
    print(f"Split{len(documents)} documents into {len(split_docs)} chunks")

# chunks=split_documents(pdf_documents)


model = SentenceTransformer("BAAI/bge-base-en-v1.5")

class embeddingmanager:
    """Handles document embedding generation using sentence transformers"""
    def __init__(self,model_name:str='BAAI/bge-base-en-v1.5'):
        """ 
        initialize the embedding manager
        Args:
            model_name: HuggingFace model name for sentence embedding"""
        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            print(f"loading embedding model:{self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Model loaded succesfully.Embedding dimension:{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"error loading model {self.model_name}:{e}")
            raise
    def generate_embedding(self,texts:List[str]):
        """ 
        Generate embedding for list of texts
        Args:
            texts: List of text strings embed
        Returns:
            numpy array of embeddings with shape (len(texts)),embedding_dim"""
        if not self.model:
            raise ValueError("model not loaded")
        print(f"Generating embedding for {len(texts)} texts...")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embedding with shape:{embeddings.shape}")
        return embeddings


## initialize the embedding manager
embeding_manager=embeddingmanager()

# documents = [
#     {
#         "content": doc.page_content,
#         "metadata": doc.metadata
#     }
#     for doc in chunks
# ]
# texts = [doc["content"] for doc in documents]
# embeddings = embeding_manager.generate_embedding(texts)


# this is sql+vector store
class VectorStore:
    """
    Large-scale FAISS Vector Store
    - Cosine similarity
    - IVF + HNSW (fast & scalable)
    - SQLite metadata storage
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_dim: int = 768,
        nlist: int = 4096,          # number of clusters
        hnsw_m: int = 32,
        embeddings: np.ndarray = None
    ):
        self.persist_directory = persist_directory
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.hnsw_m = hnsw_m

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

        # If it's already IDMap2, use it
            if isinstance(loaded_index, faiss.IndexIDMap2):
                self.index = loaded_index

            else:
            # If it has vectors, we need to wrap manually
                print("Converting existing FAISS index to IndexIDMap2...")
                d = loaded_index.d
                new_index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))  # create IDMap2

            # Re-add existing vectors
                if loaded_index.ntotal > 0:
                    vectors = np.array([loaded_index.reconstruct(i) for i in range(loaded_index.ntotal)]).astype("float32")
                    ids_old = np.arange(loaded_index.ntotal, dtype=np.int64)
                    faiss.normalize_L2(vectors)
                    new_index.add_with_ids(vectors, ids_old)

                self.index = new_index

            print("FAISS index loaded")
            self.is_trained = self.index.is_trained
            return

        if embeddings is None:
            raise ValueError("Embeddings required to create a new FAISS index")

    # --- create new index if embeddings given ---
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        num_vectors = embeddings.shape[0]
        if num_vectors < 1000:
            flat_index = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIDMap2(flat_index)
            self.index.add_with_ids(embeddings, np.arange(num_vectors, dtype=np.int64))
            self.is_trained = True
        else:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            ivf_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIDMap2(ivf_index)
            self.index.train(embeddings)
            self.index.add_with_ids(embeddings, np.arange(num_vectors, dtype=np.int64))
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

#vectorstore = VectorStore(embeddings=embeddings)  # call _init_faiss automatically

FAISS_DIR = "./data/faiss_store"

if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
    # 🔁 Load existing FAISS + SQLite
    vectorstore = VectorStore(persist_directory=FAISS_DIR, embedding_dim=768)
    print("✅ Loaded existing FAISS index")
else:
    # 🧱 First run only – build FAISS + add docs
    pdf_documents = dir_loader.load()
    chunks=split_documents(pdf_documents)
    documents = [
    {
        "content": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in chunks
    ]
    texts = [doc["content"] for doc in documents]
    embeddings = embeding_manager.generate_embedding(texts)
    vectorstore = VectorStore(
        persist_directory=FAISS_DIR,
        embedding_dim=768,
        embeddings=embeddings
    )

    vectorstore.add_documents(documents, embeddings)
    print("🚀 Built FAISS index for the first time")
# 3️⃣ Create VectorStore with embeddings
#vectorstore = VectorStore(
#    persist_directory="./data/faiss_store",
#    embedding_dim=768,
#    embeddings=embeddings
#)

# 4️⃣ Add documents safely
#vectorstore.add_documents(documents, embeddings)


class RAGRetriever:
    """
    Retrieval layer for FAISS + SQLite vector store
    """

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        nprobe: int = 10
    ):
        print(f"Retrieving documents for query: {query}")

        # 1️⃣ Convert query → embedding
        #Generate query embedding
        query_embedding=embeding_manager.generate_embedding(texts=[query])[0]
        ## not : embbedd_ query should be deined in embeddingmanager to perform this
        query_embedding = np.array([query_embedding], dtype="float32")

        # 2️⃣ FAISS similarity search
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            nprobe=nprobe
        )

        # 3️⃣ Filter by similarity score
        filtered_results = [
            r for r in results if r["score"] >= score_threshold
        ]

        print(f"Retrieved {len(filtered_results)} documents")

        return filtered_results

rag_retriever=RAGRetriever(vectorstore,embeddingmanager)

# In[26]:

## inilialize the groq llm

llm=ChatGroq(groq_api_key=groq_api_key,model_name="openai/gpt-oss-120b",temperature=0.1,max_tokens=1024)
# simple rag function: retrive context+generate respnse
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


# In[ ]:

class SerpClient:
    def __init__(self, api_key, engines=("google",)):
        self.api_key = api_key
        self.engines = engines

    def search(self, query, num=3, dedupe=True):
        all_results = []
        seen_links = set()

        for engine in self.engines:
            try:
                search = GoogleSearch({
                    "q": query,
                    "api_key": self.api_key,
                    "engine": engine,
                    "num": num,
                })
                results = search.get_dict()
            except Exception as e:
                # Fail soft in production
                print(f"[SerpClient] Error with engine '{engine}': {e}")
                continue

            if engine == "google_videos":
                videos = results.get("video_results", [])
                parsed = [
                    {
                        "title": v.get("title", ""),
                        "snippet": f'{v.get("title", "")} -- {v.get("description", "")}',
                        "link": v.get("link", ""),
                        "engine": engine,
                    }
                    for v in videos
                ]
            else:
                organic = results.get("organic_results", [])
                parsed = [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "link": r.get("link", ""),
                        "engine": engine,
                    }
                    for r in organic
                ]

            if dedupe:
                for r in parsed:
                    link = r.get("link")
                    if link and link not in seen_links:
                        seen_links.add(link)
                        all_results.append(r)
            else:
                all_results.extend(parsed)

        return all_results

    def web_search(self, query, num=3):
        return self.search(query, num=num)

    def youtube_search(self, query, num=3):
        # SerpApi uses "google_videos" for video results
        client = SerpClient(api_key=self.api_key, engines=("google_videos",))
        return client.search(query, num=num)



serp_client = SerpClient(
    api_key=serp_api_key,
    engines=("google", "youtube")
)


rag_pipeline = AdvancedRagPipeline(
    retriever=rag_retriever,
    llm=llm,
    serp_api=serp_client
)

#result = rag_pipeline.query("convex sets", stream=False)

#print(result)
print("Initialization complete. Telegram bot ready.")

# -----------------------------
# Telegram Bot
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I am your RAG bot. Ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text
    result = rag_pipeline.query(user_msg,stream=False)
    #result = await asyncio.to_thread(rag_pipeline.query, user_msg, False)
    await update.message.reply_text(result["answer"])

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    while True:
        try:
            app.run_polling(poll_interval=1, timeout=30)
        except Exception as e:
            print("Bot crashed, restarting...", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
    
   # app.run_polling(poll_interval=1, timeout=30)
