# =============================================================================
# src/pipeline/embedder.py  — v2
# =============================================================================
# Improvements over v1:
#   - Finance-aware embedding model (BAAI/bge-base-en-v1.5, matches chunker)
#   - Reads pre-computed embeddings from chunk JSON (chunker already
#     embedded everything) — NO re-embedding here, just upload to ChromaDB
#   - This makes embedder purely an I/O step, much faster

import sys
import json
import logging
import os
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_DIR = Path("data/chunks")

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]

CHROMA_HOST       = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_CHUNKS = "chunks"
BATCH_SIZE        = 200


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------
_chroma     = None
_collection = None


def get_collection():
    global _chroma, _collection
    if _collection is None:
        import chromadb
        log.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
        _chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _collection = _chroma.get_or_create_collection(
            name=COLLECTION_CHUNKS,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"Collection '{COLLECTION_CHUNKS}' ready.")
    return _collection


# ---------------------------------------------------------------------------
# Upload — no re-embedding, embeddings already exist in chunk JSON
# ---------------------------------------------------------------------------

def upload_chunks(chunks: List[dict]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    stored = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # Skip chunks missing embeddings (shouldn't happen, but safe)
        batch = [c for c in batch if c.get("embedding")]
        if not batch:
            continue

        ids        = [c["chunk_id"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        texts      = [c["text"] for c in batch]

        metadatas = []
        for c in batch:
            metadatas.append({
                "ticker"            : c["ticker"],
                "filing_id"         : c["filing_id"],
                "filing_type"       : c["filing_type"],
                "filing_date"       : c["filing_date"],
                "section_id"        : c["section_id"],
                "section_name"      : c["section_name"],
                "chunk_index"       : c["chunk_index"],
                "total_chunks"      : c["total_chunks"],
                "word_count"        : c["word_count"],
                "topic_shift_score" : c.get("topic_shift_score") or 0.0,
                "overlap_previous"  : c.get("overlap_previous", "")[:500],
                "overlap_next"      : c.get("overlap_next", "")[:500],
                "preceding_context" : c.get("preceding_context", "")[:300],
                # Filled in later by clusterer
                "cluster_id"        : -1,
                "is_cluster_head"   : False,
            })

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas,
        )
        stored += len(batch)

    return stored


def process_ticker(ticker: str) -> dict:
    chunk_dir = CHUNKS_DIR / ticker
    if not chunk_dir.exists():
        return {"ticker": ticker, "stored": 0}

    files = sorted(chunk_dir.glob("*.json"))
    if not files:
        return {"ticker": ticker, "stored": 0}

    total = 0
    for f in files:
        try:
            data   = json.load(open(f))
            chunks = data.get("chunks", [])
            if not chunks:
                continue
            stored = upload_chunks(chunks)
            total += stored
            log.info(f"  {f.name}: {stored} chunks uploaded")
        except Exception as e:
            log.error(f"  {f.name}: {e}")

    log.info(f"{ticker}: {total} chunks uploaded to ChromaDB")
    return {"ticker": ticker, "stored": total}


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS

    log.info(f"Uploading embeddings for {len(tickers)} tickers...")
    total = 0
    results = []
    for t in tickers:
        r = process_ticker(t)
        results.append(r)
        total += r["stored"]

    try:
        chroma_count = get_collection().count()
    except Exception:
        chroma_count = "unknown"

    print("\n" + "=" * 60)
    print("EMBEDDING UPLOAD COMPLETE")
    print("=" * 60)
    print(f"  Chunks uploaded this run : {total}")
    print(f"  Total in ChromaDB        : {chroma_count}")
    for r in results:
        if r["stored"] > 0:
            print(f"    {r['ticker']:<8} {r['stored']} chunks")


if __name__ == "__main__":
    main()