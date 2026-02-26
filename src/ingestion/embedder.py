import openai
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LegalEmbedder:
    """
    Generates embeddings for legal text chunks.
    Supports OpenAI and local sentence-transformer models.
    Includes batching and retry logic for production use.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        use_local: bool = False,
        local_model: str = "intfloat/multilingual-e5-large"
    ):
        self.model = model
        self.batch_size = batch_size
        self.use_local = use_local
        
        if use_local:
            # Use multilingual E5 for better Hindi/regional language support
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(local_model)
            logger.info(f"Loaded local model: {local_model}")
        else:
            self.client = openai.OpenAI()
    
    def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API with simple retry logic."""
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                last_err = e
                wait = 4 * (2 ** attempt)  # 4s, 8s, 16s
                logger.warning(f"OpenAI embedding attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        raise last_err
    
    def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        """Embed using local sentence-transformer model."""
        # Prefix for E5 model (instruction-tuned)
        prefixed = [f"passage: {text}" for text in texts]
        embeddings = self.local_model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, handling batching automatically."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if self.use_local:
                batch_embeddings = self._embed_batch_local(batch)
            else:
                batch_embeddings = self._embed_batch_openai(batch)
            
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting cushion
            if not self.use_local and i + self.batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (uses query prefix for E5 models)."""
        if self.use_local:
            query_prefixed = f"query: {query}"
            return self.local_model.encode(
                [query_prefixed],
                normalize_embeddings=True
            )[0].tolist()
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=[query]
            )
            return response.data[0].embedding
