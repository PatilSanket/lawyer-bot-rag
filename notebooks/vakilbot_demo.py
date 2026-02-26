# VakilBot Interactive Demo
# Run this notebook after completing ingestion to explore the system.
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. Copy .env.example to .env and fill in credentials
#   3. Run src/ingestion/run_ingestion.py to index legal documents
#
# Then open this notebook with Jupyter:
#   jupyter notebook notebooks/vakilbot_demo.ipynb

import sys
sys.path.insert(0, "../src")

from dotenv import load_dotenv
load_dotenv("../.env")

import os
from elasticsearch import Elasticsearch
from ingestion.embedder import LegalEmbedder
from retrieval.searcher import LegalSearcher
from rag.vakilbot import VakilBot

# Initialize
es = Elasticsearch(os.getenv("ELASTIC_URL"), api_key=os.getenv("ELASTIC_API_KEY"))
embedder = LegalEmbedder()
searcher = LegalSearcher(es, os.getenv("INDEX_NAME"), embedder)
bot = VakilBot(searcher)

# Example query
query = "What is the punishment for murder under IPC?"
answer = bot.answer(query, stream=False)
print(answer)
