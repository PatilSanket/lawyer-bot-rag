# src/api/main.py
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

from retrieval.searcher import LegalSearcher
from ingestion.embedder import LegalEmbedder
from rag.vakilbot import VakilBot

load_dotenv()

app = FastAPI(title="VakilBot API", version="1.0.0")

# Initialize components at startup
# Supports both API key (cloud) and username/password (local Docker)
_es_kwargs = {"hosts": [os.getenv("ELASTIC_URL", "http://localhost:9200")]}
if os.getenv("ELASTIC_API_KEY"):
    _es_kwargs["api_key"] = os.getenv("ELASTIC_API_KEY")
else:
    _es_kwargs["basic_auth"] = (
        os.getenv("ELASTIC_USERNAME", "elastic"),
        os.getenv("ELASTIC_PASSWORD", "vakilbot123"),
    )

es = Elasticsearch(**_es_kwargs)

embedder = LegalEmbedder()
searcher = LegalSearcher(es, os.getenv("INDEX_NAME"), embedder)
bot = VakilBot(searcher)

class QueryRequest(BaseModel):
    question: str
    act_filter: Optional[str] = None
    stream: bool = True

@app.post("/ask")
async def ask(request: QueryRequest):
    """Ask VakilBot a legal question."""
    if request.stream:
        return StreamingResponse(
            bot.answer(request.question, act_filter=request.act_filter, stream=True),
            media_type="text/event-stream"
        )
    else:
        answer = bot.answer(request.question, act_filter=request.act_filter, stream=False)
        return {"answer": answer}

@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    act: Optional[str] = Query(None, description="Filter by act name"),
    k: int = Query(5, description="Number of results")
):
    """Raw semantic search endpoint for debugging/exploration."""
    results = searcher.hybrid_search(q, k=k, act_filter=act)
    return {"results": results, "count": len(results)}

@app.get("/health")
async def health():
    count = es.count(index=os.getenv("INDEX_NAME"))["count"]
    return {"status": "healthy", "documents_indexed": count}
