# src/ingestion/indexer.py
from elasticsearch import Elasticsearch, helpers
from datetime import datetime, timezone
import os
from tqdm import tqdm
from ingestion.chunker import Chunk
from ingestion.embedder import LegalEmbedder

class LegalIndexer:
    """
    Indexes legal document chunks into Elasticsearch.
    Handles bulk indexing with progress tracking.
    """
    
    def __init__(self, es_client: Elasticsearch, index_name: str, embedder: LegalEmbedder):
        self.es = es_client
        self.index_name = index_name
        self.embedder = embedder
    
    def index_chunks(self, chunks: list[Chunk], batch_size: int = 50):
        """
        Index chunks in batches.
        For each batch: generate embeddings -> build ES actions -> bulk index.
        """
        total = len(chunks)
        indexed = 0
        failed = 0
        
        with tqdm(total=total, desc="Indexing legal chunks") as pbar:
            for i in range(0, total, batch_size):
                batch = chunks[i:i + batch_size]
                
                # Generate embeddings for batch
                texts = [chunk.content for chunk in batch]
                try:
                    embeddings = self.embedder.embed_texts(texts)
                except Exception as e:
                    print(f"Embedding error on batch {i}: {e}")
                    failed += len(batch)
                    pbar.update(len(batch))
                    continue
                
                # Build bulk actions
                actions = []
                for chunk, embedding in zip(batch, embeddings):
                    doc = {
                        "_index": self.index_name,
                        "_id": chunk.chunk_id,
                        "_source": {
                            **chunk.to_dict(),
                            "content_embedding": embedding,
                            "indexed_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                    actions.append(doc)
                
                # Bulk index
                success, errors = helpers.bulk(
                    self.es,
                    actions,
                    raise_on_error=False,
                    stats_only=False
                )
                
                indexed += success
                if errors:
                    failed += len(errors)
                    for err in errors[:3]:  # Log first 3 errors
                        print(f"Index error: {err}")
                
                pbar.update(len(batch))
        
        print(f"\n✅ Indexing complete: {indexed} chunks indexed, {failed} failed")
        return indexed, failed
