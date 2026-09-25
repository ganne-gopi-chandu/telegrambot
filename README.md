# Telegram RAG Student Assistance Bot 🤖📚

An AI-powered Telegram Bot utilizing Retrieval-Augmented Generation (RAG), Groq LLM inference, FAISS vector storage, and Google SerpApi search to answer questions on Convex Optimization (or custom textbook PDFs).

---

## 🌟 Features

- **RAG Architecture**: Ingests textbook PDFs (e.g. Convex Optimization by Stephen Boyd) and retrieves relevant excerpts using FAISS vector search & sentence embeddings.
- **Smart Routing & Tool Usage**: Routes user queries to internal knowledge retrieval or real-time web search (via SerpApi).
- **Fast Inference**: Powered by Groq AI models.
- **Telegram Integration**: Asynchronous Telegram bot built with `python-telegram-bot`.

---

## 📁 Repository Structure

```
telegrambot/
├── convexopti.pdf         # Sample textbook PDF for ingestion
├── .env.example           # Template for environment configuration
├── requirements.txt       # Python dependencies
├── separatecodes/         # Modular production scripts
│   ├── bot.py             # Main Telegram Bot runner
│   ├── ingest.py          # Document loader & FAISS vectorstore generator
│   ├── rag.py             # RAG pipeline interface
│   └── rag_backend.py     # Embeddings, vector store, and search retriever
└── README.md
```

---

## ⚡ Quick Start & Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telegrambot.git
cd telegrambot
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
SERP_API_KEY=your_serp_api_key
```

### 4. Run Ingestion

Ingest the textbook PDF to generate the local FAISS vector store index:

```bash
python separatecodes/ingest.py
```

### 5. Launch the Bot

Start the Telegram bot:

```bash
python separatecodes/bot.py
```

Now start a chat with your Telegram Bot and ask any question! 🚀
