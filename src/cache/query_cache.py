# src/cache/query_cache.py
import hashlib
import json
from functools import wraps
from typing import Optional
import redis

class QueryCache:
    """
    Redis-based cache for query results.
    Legal queries often repeat — users asking about divorce rights, 
    property disputes, etc. Cache dramatically reduces latency and cost.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl  # 1 hour default
    
    def _hash_query(self, query: str, filters: dict) -> str:
        key_data = json.dumps({"query": query.lower().strip(), "filters": filters}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, filters: dict = {}) -> Optional[list]:
        key = self._hash_query(query, filters)
        cached = self.redis.get(f"vakil:search:{key}")
        return json.loads(cached) if cached else None
    
    def set(self, query: str, results: list, filters: dict = {}):
        key = self._hash_query(query, filters)
        self.redis.setex(
            f"vakil:search:{key}",
            self.ttl,
            json.dumps(results)
        )
