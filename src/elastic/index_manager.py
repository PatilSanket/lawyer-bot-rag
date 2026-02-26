# src/elastic/index_manager.py
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
import os

def create_legal_index(es: Elasticsearch, index_name: str):
    """
    Creates an Elasticsearch index optimized for Indian legal document search.
    Uses a multi-vector strategy: dense (OpenAI), sparse (ELSER), and BM25.
    """
    
    mapping = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
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
                        # Remove noise words common in legal text
                        "stopwords": ["the", "of", "and", "or", "in", "to", "a", "an",
                                      "hereinafter", "aforesaid", "aforementioned",
                                      "notwithstanding", "pursuant"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # === TEXT FIELDS (BM25) ===
                "content": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"},  # for exact match
                        "suggest": {                      # for autocomplete
                            "type": "completion"
                        }
                    }
                },
                "section_title": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                    "boost": 2.0  # section titles get 2x weight in BM25
                },
                
                # === DENSE VECTOR (kNN / Semantic) ===
                "content_embedding": {
                    "type": "dense_vector",
                    "dims": 1024,          # intfloat/multilingual-e5-large
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,           # HNSW connections per node (higher = more accurate)
                        "ef_construction": 100  # construction-time accuracy
                    }
                },
                
                # === SPARSE VECTOR (ELSER) ===
                "content_sparse": {
                    "type": "sparse_vector"
                },
                
                # === METADATA FIELDS (Filters) ===
                "act_name": {
                    "type": "keyword"
                },
                "act_year": {
                    "type": "integer"
                },
                "chapter": {
                    "type": "keyword"
                },
                "section_number": {
                    "type": "keyword"
                },
                "doc_type": {
                    "type": "keyword"  # statute | judgment | commentary
                },
                "tags": {
                    "type": "keyword"  # criminal, civil, corporate, etc.
                },
                "source_file": {
                    "type": "keyword"
                },
                
                # === TIMESTAMP ===
                "indexed_at": {
                    "type": "date"
                }
            }
        }
    }
    
    try:
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists. Skipping creation.")
            return
        
        es.indices.create(index=index_name, body=mapping)
        print(f"✅ Index '{index_name}' created successfully.")
        
    except RequestError as e:
        print(f"❌ Error creating index: {e.info}")
        raise
