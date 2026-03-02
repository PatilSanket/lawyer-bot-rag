# src/elastic/index_manager.py
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
import os

def create_legal_index(es: Elasticsearch, index_name: str):
    """
    Creates an Elasticsearch index optimized for Indian legal document search.
    Uses dense vector (kNN semantic) + BM25 hybrid search.
    """

    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,   # single-node local dev — 0 keeps index green
        "analysis": {
            "analyzer": {
                "legal_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "legal_stop_words",
                        "snowball"
                    ]
                }
            },
            "filter": {
                "legal_stop_words": {
                    "type": "stop",
                    "stopwords": [
                        "the", "of", "and", "or", "in", "to", "a", "an",
                        "hereinafter", "aforesaid", "aforementioned",
                        "notwithstanding", "pursuant"
                    ]
                }
            }
        }
    }

    mappings = {
        "properties": {
            # === TEXT FIELDS (BM25) ===
            "content": {
                "type": "text",
                "analyzer": "legal_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"},   # for exact match
                }
            },
            # boost param removed in ES 8.x — use query-time boosting instead
            "section_title": {
                "type": "text",
                "analyzer": "legal_analyzer"
            },

            # === DENSE VECTOR (kNN / Semantic) ===
            "content_embedding": {
                "type": "dense_vector",
                "dims": 384,            # all-MiniLM-L6-v2 output size
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "hnsw",
                    "m": 16,
                    "ef_construction": 100
                }
            },

            # === METADATA FIELDS (Filters) ===
            "act_name":       {"type": "keyword"},
            "act_year":       {"type": "integer"},
            "chapter":        {"type": "keyword"},
            "section_number": {"type": "keyword"},
            "doc_type":       {"type": "keyword"},
            "tags":           {"type": "keyword"},
            "source_file":    {"type": "keyword"},

            # === TIMESTAMP ===
            "indexed_at": {"type": "date"}
        }
    }

    try:
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists. Skipping creation.")
            return

        es.indices.create(index=index_name, settings=settings, mappings=mappings)
        print(f"✅ Index '{index_name}' created successfully.")

    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise
