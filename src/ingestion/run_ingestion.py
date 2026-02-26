# src/ingestion/run_ingestion.py
import hashlib
import json
import os
import pickle
from pathlib import Path


# ---------------------------------------------------------------------------
# Disk-based Embedding Cache
# ---------------------------------------------------------------------------
# Saves embeddings to data/embeddings_cache.pkl, keyed by SHA-256 of the
# chunk text. On re-runs only NEW or CHANGED chunks hit the OpenAI API.

CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "embeddings_cache.pkl"


def _load_cache() -> dict:
    """Load {text_hash -> embedding} from disk, or return empty dict."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"📦 Embedding cache loaded: {len(cache)} entries from {CACHE_PATH}")
        return cache
    return {}


def _save_cache(cache: dict):
    """Persist the embedding cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"💾 Embedding cache saved: {len(cache)} entries → {CACHE_PATH}")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_with_cache(texts: list[str], embedder, cache: dict) -> list[list[float]]:
    """
    Returns embeddings for all texts.
    - Cache HIT  → returns stored embedding, no API call
    - Cache MISS → batches new texts, calls OpenAI, stores results
    """
    results = [None] * len(texts)
    miss_indices = []
    miss_texts = []

    for i, text in enumerate(texts):
        h = _text_hash(text)
        if h in cache:
            results[i] = cache[h]
        else:
            miss_indices.append(i)
            miss_texts.append(text)

    if miss_texts:
        print(f"   🔄 Embedding {len(miss_texts)} new chunks (OpenAI API)  "
              f"[{len(texts) - len(miss_texts)} from cache]")
        new_embeddings = embedder.embed_texts(miss_texts)
        for idx, text, emb in zip(miss_indices, miss_texts, new_embeddings):
            cache[_text_hash(text)] = emb
            results[idx] = emb
    else:
        print(f"   ✅ All {len(texts)} chunks served from cache — no API call needed")

    return results


# ---------------------------------------------------------------------------
# Master Ingestion Script
# ---------------------------------------------------------------------------

def run_full_ingestion():
    """
    End-to-end ingestion pipeline for the Indian legal corpus.
    Embeddings are cached to disk so incremental re-runs are cheap.
    """
    from elasticsearch import Elasticsearch
    from dotenv import load_dotenv

    from elastic.index_manager import create_legal_index
    from ingestion.parser import IndianLegalParser
    from ingestion.chunker import LegalChunker
    from ingestion.embedder import LegalEmbedder
    from ingestion.indexer import LegalIndexer

    load_dotenv()

    # Connect to Elasticsearch (supports local Docker + Elastic Cloud)
    _es_kwargs = {"hosts": [os.getenv("ELASTIC_URL", "http://localhost:9200")]}
    if os.getenv("ELASTIC_API_KEY"):
        _es_kwargs["api_key"] = os.getenv("ELASTIC_API_KEY")
    else:
        _es_kwargs["basic_auth"] = (
            os.getenv("ELASTIC_USERNAME", "elastic"),
            os.getenv("ELASTIC_PASSWORD", "vakilbot123"),
        )

    es = Elasticsearch(**_es_kwargs)
    print("Connected to Elasticsearch:", es.info()["version"]["number"])

    index_name = os.getenv("INDEX_NAME", "indian-legal-corpus")

    # Step 1: Create index
    create_legal_index(es, index_name)

    # Step 2: Initialize components
    parser = IndianLegalParser()
    chunker = LegalChunker(max_tokens=512, overlap_tokens=64)
    embedder = LegalEmbedder(
        use_local=True,
        local_model="intfloat/multilingual-e5-large"  # ~1.2 GB, downloads once
    )
    indexer = LegalIndexer(es, index_name, embedder)

    # Step 3: Load embedding cache from disk
    embedding_cache = _load_cache()

    # Step 4: Process each legal document
    pdf_directory = Path(__file__).resolve().parents[2] / "data" / "legal_pdfs"
    act_mapping = {
        "bns-2023.pdf": "Bharatiya Nyay Sanhita, 2023",
        "bnss-2023.pdf": "Bharatiya Nagrik Suraksha Sanhita, 2023",
        "companies-act-2013.pdf": "Companies Act, 2013",
        "constitution.pdf": "The Constitution of India, 1950",
        "consumer-protection-act-2019.pdf": "Consumer Protection Act, 2019",
        "it-act-2020.pdf": "Information Technology Act, 2020",
        "pocso-act-2012.pdf": "Protection of Children from Sexual Offences Act, 2012",
        "labour-laws-2025.pdf": "Labour Laws, 2025",
        "women-protection-2005.pdf": "Women Protection Act, 2005",
    }

    all_chunks = []

    for pdf_file, act_name in act_mapping.items():
        pdf_path = pdf_directory / pdf_file
        if not pdf_path.exists():
            print(f"⚠️  Skipping {pdf_file} - not found")
            continue

        print(f"\n📚 Processing: {act_name}")

        documents = parser.parse_pdf(str(pdf_path), act_name)
        print(f"   Parsed {len(documents)} sections")

        chunks = list(chunker.chunk_corpus(documents))
        print(f"   Generated {len(chunks)} chunks")

        all_chunks.extend(chunks)

    print(f"\n📦 Total chunks to index: {len(all_chunks)}")

    # Step 5: Embed with cache (only new chunks hit OpenAI)
    print("\n🔢 Generating embeddings...")
    texts = [chunk.content for chunk in all_chunks]
    embeddings = embed_with_cache(texts, embedder, embedding_cache)

    # Save updated cache to disk immediately after embedding
    _save_cache(embedding_cache)

    # Step 6: Index into Elasticsearch (inject pre-computed embeddings)
    print("\n📤 Indexing into Elasticsearch...")
    from elasticsearch import helpers
    from datetime import datetime, timezone

    actions = []
    for chunk, embedding in zip(all_chunks, embeddings):
        actions.append({
            "_index": index_name,
            "_id": chunk.chunk_id,
            "_source": {
                **chunk.to_dict(),
                "content_embedding": embedding,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    success, errors = helpers.bulk(es, actions, raise_on_error=False, stats_only=False)
    print(f"✅ Indexed {success} chunks" + (f", {len(errors)} errors" if errors else ""))

    # Step 7: Verify
    count = es.count(index=index_name)["count"]
    print(f"\n🎉 Done! {count} documents in index '{index_name}'")
    print(f"💡 Embedding cache at: {CACHE_PATH}  ({len(embedding_cache)} entries)")


if __name__ == "__main__":
    run_full_ingestion()
