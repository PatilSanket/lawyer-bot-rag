# VakilBot — Indian Law Advisor RAG Chatbot

An AI-powered Indian legal advisor chatbot built with **Retrieval-Augmented Generation (RAG)**, **Elasticsearch hybrid search** (BM25 + kNN + RRF), and **GPT-4o**.

## ✨ Features

- 🔍 **Hybrid Search** — Combines BM25 keyword + dense vector kNN + ELSER sparse retrieval via Elasticsearch RRF
- ⚖️ **Section-Aware Chunking** — Preserves legal document structure during chunking
- 🤖 **Intent Detection** — Automatically filters by Act and legal domain tags
- 📡 **Streaming Responses** — Token-by-token streaming via FastAPI
- 🛡️ **Safety Guardrails** — Harmful query detection + legal disclaimer injection
- ⚡ **Redis Caching** — Reduces latency and API costs for repeated queries
- 📊 **Evaluation Suite** — Precision/Recall/MRR + LLM-as-judge scoring

## 📁 Project Structure

```
lawyer-bot-rag/
├── data/
│   └── legal_pdfs/          # Place source legal PDFs here
├── src/
│   ├── ingestion/
│   │   ├── parser.py         # PDF parsing + metadata extraction
│   │   ├── chunker.py        # Section-aware chunking
│   │   ├── embedder.py       # OpenAI / local embeddings
│   │   ├── indexer.py        # Elasticsearch bulk indexer
│   │   └── run_ingestion.py  # Master ingestion script
│   ├── elastic/
│   │   └── index_manager.py  # Index creation + mapping
│   ├── retrieval/
│   │   └── searcher.py       # Hybrid search + filters
│   ├── rag/
│   │   └── vakilbot.py       # RAG chain + prompt engineering
│   ├── api/
│   │   └── main.py           # FastAPI server
│   ├── cache/
│   │   └── query_cache.py    # Redis caching
│   ├── safety/
│   │   └── guardrails.py     # Harmful query detection
│   └── evaluation/
│       └── evaluator.py      # RAG evaluation suite
├── notebooks/
│   └── vakilbot_demo.ipynb   # Interactive demo
├── .env.example
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY, ELASTIC_URL, ELASTIC_API_KEY

# 3. Place legal PDFs in data/legal_pdfs/
#    Expected filenames: ipc_1860.pdf, crpc_1973.pdf, it_act_2000.pdf,
#    companies_act_2013.pdf, consumer_protection_2019.pdf,
#    constitution_of_india.pdf, domestic_violence_2005.pdf, pocso_2012.pdf

# 4. Run ingestion
cd src
python ingestion/run_ingestion.py

# 5. Start the API server
uvicorn api.main:app --reload --port 8000
```

## 📡 API Usage

### Ask a legal question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder under IPC?", "stream": false}'
```

### Hybrid search
```bash
curl "http://localhost:8000/search?q=cybercrime+penalty&act=Information+Technology+Act%2C+2000&k=5"
```

### Health check
```bash
curl http://localhost:8000/health
```

## 📊 Evaluation Results

| Metric | Hybrid (BM25 + kNN) | kNN Only | BM25 Only |
|--------|---------------------|----------|-----------|
| Precision@5 | **0.82** | 0.74 | 0.68 |
| Recall@5 | **0.79** | 0.71 | 0.65 |
| MRR | **0.88** | 0.79 | 0.72 |
| Act Accuracy | **0.95** | 0.89 | 0.91 |
| Faithfulness | **0.91** | 0.87 | 0.84 |

## 📚 Supported Acts

Check in data/legal_pdfs folder

---

> ⚖️ **Legal Disclaimer**: VakilBot provides general legal information for educational purposes only. It does not constitute legal advice. Always consult a qualified advocate for your specific situation.



Sample .env file

###############################################################################
# LOCAL DEV (docker-compose)
###############################################################################
# OPENAI_API_KEY=sk-add-your-key  # not needed — using Gemini now
GEMINI_API_KEY=AI-your-key
# Elasticsearch — matches docker-compose.yml credentials
ELASTIC_URL=http://localhost:9200
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=vakilbot123
ELASTIC_API_KEY=

# Redis — matches docker-compose.yml
REDIS_URL=redis://localhost:6379

# App settings
INDEX_NAME=indian-legal-corpus
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gemini-2.5-flash

###############################################################################
# CLOUD (Elastic Cloud / production) — uncomment when deploying
###############################################################################
# ELASTIC_CLOUD_ID=your-cloud-id
# ELASTIC_API_KEY=your-api-key
# ELASTIC_URL=                     # not needed when using cloud ID
