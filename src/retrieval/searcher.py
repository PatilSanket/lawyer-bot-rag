# src/retrieval/searcher.py
from elasticsearch import Elasticsearch
from typing import Optional
import os

class LegalSearcher:
    """
    Production-grade hybrid search for Indian legal documents.
    Combines:
    - Dense kNN (semantic similarity via embeddings)
    - BM25 (keyword/term matching)
    - Metadata filters (act, year, doc_type, tags)
    - RRF fusion for combining rankings
    """
    
    def __init__(self, es: Elasticsearch, index_name: str, embedder):
        self.es = es
        self.index_name = index_name
        self.embedder = embedder
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        act_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        tag_filter: Optional[list[str]] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> list[dict]:
        """
        Execute hybrid search combining kNN + BM25 with optional filters.
        Uses Elasticsearch's native RRF (Reciprocal Rank Fusion).
        """
        
        # Generate query embedding
        query_vector = self.embedder.embed_query(query)
        
        # Build filter clauses
        filters = self._build_filters(act_filter, doc_type_filter, tag_filter, year_range)
        
        # Hybrid query using RRF
        query_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^1.0",
                                    "section_title^2.0"  # boost section title matches
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "filter": filters
                }
            },
            "knn": {
                "field": "content_embedding",
                "query_vector": query_vector,
                "k": k * 2,               # retrieve more candidates for RRF
                "num_candidates": k * 10,  # HNSW search breadth
                "filter": filters
            },
            "rank": {
                "rrf": {
                    "window_size": k * 3,
                    "rank_constant": 60    # smoothing constant
                }
            },
            "_source": {
                "excludes": ["content_embedding", "content_sparse"]  # don't return vectors
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 2,
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"]
                    }
                }
            }
        }
        
        response = self.es.search(index=self.index_name, body=query_body)
        
        return self._format_results(response)
    
    def semantic_only_search(self, query: str, k: int = 5) -> list[dict]:
        """Pure vector search — useful for benchmarking."""
        query_vector = self.embedder.embed_query(query)
        
        response = self.es.search(
            index=self.index_name,
            body={
                "knn": {
                    "field": "content_embedding",
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": k * 10
                },
                "size": k,
                "_source": {"excludes": ["content_embedding"]}
            }
        )
        return self._format_results(response)
    
    def section_lookup(self, act_name: str, section_number: str) -> Optional[dict]:
        """Direct lookup by act + section number (exact retrieval)."""
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"act_name": act_name}},
                            {"term": {"section_number": section_number}}
                        ]
                    }
                },
                "size": 5
            }
        )
        
        hits = response["hits"]["hits"]
        return hits[0]["_source"] if hits else None
    
    def _build_filters(self, act_filter, doc_type_filter, tag_filter, year_range):
        filters = []
        
        if act_filter:
            filters.append({"term": {"act_name": act_filter}})
        
        if doc_type_filter:
            filters.append({"term": {"doc_type": doc_type_filter}})
        
        if tag_filter:
            filters.append({"terms": {"tags": tag_filter}})
        
        if year_range:
            filters.append({
                "range": {
                    "act_year": {
                        "gte": year_range[0],
                        "lte": year_range[1]
                    }
                }
            })
        
        return filters
    
    def _format_results(self, response: dict) -> list[dict]:
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            highlights = hit.get("highlight", {}).get("content", [])
            
            results.append({
                "score": hit["_score"],
                "chunk_id": source.get("chunk_id"),
                "content": source.get("content"),
                "act_name": source.get("act_name"),
                "act_year": source.get("act_year"),
                "section_number": source.get("section_number"),
                "section_title": source.get("section_title"),
                "chapter": source.get("chapter"),
                "doc_type": source.get("doc_type"),
                "tags": source.get("tags", []),
                "highlights": highlights
            })
        
        return results
