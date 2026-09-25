# rag.py
import os
try:
    from rag_backend import VectorStore, RAGRetriever, AdvancedRagPipeline, embedding_manager, SerpClient
except ImportError:
    from separatecodes.rag_backend import VectorStore, RAGRetriever, AdvancedRagPipeline, embedding_manager, SerpClient
from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(BASE_DIR, "data", "faiss_store")

vectorstore = VectorStore(persist_directory=STORE_DIR, embedding_dim=768)
retriever = RAGRetriever(vectorstore, embedding_manager)

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024
)

serp_client = SerpClient(api_key=os.environ["SERP_API_KEY"])

rag = AdvancedRagPipeline(retriever, llm, serp_client)
def answer_question(q: str) -> str:
    r = rag.query(q)
    return f"🧭 Route: {r['route']}\n📊 Tool confidence: {r['tool_confidence']}\n\n💡 {r['answer']}"

if __name__ == "__main__":
    print(answer_question("convex sets"))
