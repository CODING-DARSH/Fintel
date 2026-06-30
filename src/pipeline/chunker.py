# =============================================================================
# src/pipeline/chunker.py  — v2
# =============================================================================
# Improvements over v1:
#   - Dynamic paragraph grouping by word count (not fixed 4 sentences)
#   - Per-section similarity thresholds (Risk Factors tighter than MD&A)
#   - No re-embedding for shift_score (reuse paragraph embeddings)
#   - Chunk embedding stored in JSON (embedder reads it, no recompute)
#   - Preceding context stored per chunk (for agent reasoning)

import sys
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR    = Path("data/chunks")

MIN_WORDS             = 100
MAX_WORDS             = 800
OVERLAP_SENTENCES     = 2
PARA_TARGET_WORDS     = 150   # dynamic grouping target (not fixed sentence count)

# Per-section cosine similarity thresholds for topic shift detection.
# Lower = more splits (finer granularity)
# Higher = fewer splits (keeps narrative together)
SECTION_THRESHOLDS = {
    "1"   : 0.55,  # Business — moderate narrative
    "1A"  : 0.40,  # Risk Factors — many distinct risks, split finely
    "1B"  : 0.65,  # Very short, keep together
    "2"   : 0.55,  # Properties
    "3"   : 0.50,  # Legal proceedings
    "5"   : 0.55,  # Market info
    "7"   : 0.62,  # MD&A — continuous narrative, fewer splits
    "7A"  : 0.50,  # Quantitative disclosures
    "8"   : 0.68,  # Financial statements — keep tables together
    "9"   : 0.62,
    "9A"  : 0.55,
    # 8-K items — typically short, keep together
    "1.01": 0.62,
    "1.02": 0.62,
    "2.01": 0.62,
    "2.02": 0.62,
    "2.03": 0.60,
    "2.05": 0.62,
    "5.02": 0.62,
    "7.01": 0.58,
    "8.01": 0.58,
    "9.01": 0.70,  # exhibits — one big block
}
DEFAULT_THRESHOLD = 0.50

SECTION_NAMES = {
    "1"   : "Business",
    "1A"  : "Risk Factors",
    "1B"  : "Unresolved Staff Comments",
    "2"   : "Properties",
    "3"   : "Legal Proceedings",
    "5"   : "Market for Registrant",
    "7"   : "MD&A",
    "7A"  : "Quantitative Disclosures",
    "8"   : "Financial Statements",
    "9"   : "Changes in Accountants",
    "9A"  : "Controls and Procedures",
    "1.01": "Entry into Material Agreement",
    "1.02": "Termination of Agreement",
    "2.01": "Completion of Acquisition",
    "2.02": "Results of Operations",
    "2.03": "Creation of Financial Obligation",
    "2.05": "Departure of Officers",
    "5.02": "Departure/Appointment of Officers",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]


# ---------------------------------------------------------------------------
# Model — loaded once
# ---------------------------------------------------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading BAAI/bge-base-en-v1.5...")
        # Finance-aware, 768-dim, better than MiniLM for domain text
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        log.info("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\(\"\'])", text)
    return [s.strip() for s in sentences if s.strip()]


def dynamic_paragraphs(text: str, target_words: int = PARA_TARGET_WORDS) -> List[str]:
    """
    Group sentences into pseudo-paragraphs targeting ~target_words each.
    Dynamic: long sentences → fewer per group, short → more per group.
    This eliminates the fixed-4-sentences heuristic.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    paragraphs  = []
    current     = []
    current_wc  = 0

    for sent in sentences:
        sent_wc = len(sent.split())
        # If adding this sentence exceeds target AND we have content, flush
        if current_wc + sent_wc > target_words and current:
            paragraphs.append(" ".join(current))
            current    = [sent]
            current_wc = sent_wc
        else:
            current.append(sent)
            current_wc += sent_wc

    if current:
        paragraphs.append(" ".join(current))

    return [p for p in paragraphs if p.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def last_n_sentences(text: str, n: int) -> str:
    sents = split_sentences(text)
    return " ".join(sents[-n:]) if sents else ""


def first_n_sentences(text: str, n: int) -> str:
    sents = split_sentences(text)
    return " ".join(sents[:n]) if sents else ""


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def semantic_chunk_section(
    section_text : str,
    ticker       : str,
    filing_id    : str,
    filing_type  : str,
    filing_date  : str,
    section_id   : str,
) -> List[dict]:

    if not section_text or word_count(section_text) < 20:
        return []

    model     = get_model()
    threshold = SECTION_THRESHOLDS.get(section_id, DEFAULT_THRESHOLD)

    # Single chunk for very short sections
    if word_count(section_text) <= MIN_WORDS:
        emb = model.encode([section_text], show_progress_bar=False)[0]
        return [{
            "chunk_id"          : f"{filing_id}_{section_id}_001",
            "ticker"            : ticker,
            "filing_id"         : filing_id,
            "filing_type"       : filing_type,
            "filing_date"       : filing_date,
            "section_id"        : section_id,
            "section_name"      : SECTION_NAMES.get(section_id, section_id),
            "chunk_index"       : 1,
            "total_chunks"      : 1,
            "text"              : section_text.strip(),
            "overlap_previous"  : "",
            "overlap_next"      : "",
            "preceding_context" : "",
            "word_count"        : word_count(section_text),
            "topic_shift_score" : None,
            "embedding"         : emb.tolist(),
        }]

    # Step 1: dynamic paragraph grouping
    paragraphs = dynamic_paragraphs(section_text)
    if not paragraphs:
        return []

    # Step 2: embed all paragraphs in one batch
    # Reuse these embeddings for BOTH topic-shift detection AND shift_score.
    # No double-embedding.
    para_embeddings = model.encode(paragraphs, show_progress_bar=False)

    # Step 3: find topic shift boundaries using per-section threshold
    boundaries = [0]
    for i in range(len(paragraphs) - 1):
        sim = cosine_sim(para_embeddings[i], para_embeddings[i + 1])
        if sim < threshold:
            boundaries.append(i + 1)
    boundaries.append(len(paragraphs))

    # Step 4: group paragraphs into raw chunks
    raw_chunks: List[Tuple[str, List[np.ndarray]]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        text  = " ".join(paragraphs[start:end])
        embs  = para_embeddings[start:end]
        raw_chunks.append((text, embs))

    # Step 5: size guardrails — merge small, split large
    merged: List[Tuple[str, np.ndarray]] = []
    buf_text = ""
    buf_embs = []

    for text, embs in raw_chunks:
        if word_count(buf_text) + word_count(text) < MIN_WORDS:
            buf_text = (buf_text + " " + text).strip()
            buf_embs.extend(embs)
        else:
            if buf_text:
                centroid = np.mean(buf_embs, axis=0) if buf_embs else embs[0]
                merged.append((buf_text, centroid))
            buf_text = text
            buf_embs = list(embs)

    if buf_text:
        centroid = np.mean(buf_embs, axis=0) if buf_embs else np.zeros(para_embeddings.shape[1])
        merged.append((buf_text, centroid))

    # Split oversized chunks at sentence boundary
    final: List[Tuple[str, np.ndarray]] = []
    for text, centroid_emb in merged:
        if word_count(text) <= MAX_WORDS:
            final.append((text, centroid_emb))
        else:
            sents   = split_sentences(text)
            current = ""
            for sent in sents:
                if word_count(current) + word_count(sent) > MAX_WORDS:
                    if current:
                        # Re-embed this split piece
                        emb = model.encode([current], show_progress_bar=False)[0]
                        final.append((current.strip(), emb))
                    current = sent
                else:
                    current = (current + " " + sent).strip()
            if current:
                emb = model.encode([current], show_progress_bar=False)[0]
                final.append((current.strip(), emb))

    if not final:
        return []

    # Step 6: build chunk dicts
    total   = len(final)
    results = []
    for idx, (text, chunk_emb) in enumerate(final):
        chunk_num = idx + 1
        chunk_id  = f"{filing_id}_{section_id}_{chunk_num:03d}"

        # Overlap — carry sentences between chunks
        overlap_prev = last_n_sentences(final[idx - 1][0], OVERLAP_SENTENCES) if idx > 0 else ""
        overlap_next = first_n_sentences(final[idx + 1][0], OVERLAP_SENTENCES) if idx < total - 1 else ""

        # Preceding context — summary of what came before (first 200 chars of previous chunk)
        preceding = final[idx - 1][0][:200] + "..." if idx > 0 else ""

        # Topic shift score: dissimilarity to previous chunk (reuse stored embeddings)
        shift_score = None
        if idx > 0:
            shift_score = round(1.0 - cosine_sim(chunk_emb, final[idx - 1][1]), 3)

        results.append({
            "chunk_id"          : chunk_id,
            "ticker"            : ticker,
            "filing_id"         : filing_id,
            "filing_type"       : filing_type,
            "filing_date"       : filing_date,
            "section_id"        : section_id,
            "section_name"      : SECTION_NAMES.get(section_id, section_id),
            "chunk_index"       : chunk_num,
            "total_chunks"      : total,
            "text"              : text.strip(),
            "overlap_previous"  : overlap_prev,
            "overlap_next"      : overlap_next,
            "preceding_context" : preceding,
            "word_count"        : word_count(text),
            "topic_shift_score" : shift_score,
            "embedding"         : chunk_emb.tolist(),   # stored → embedder reads directly
        })

    return results


# ---------------------------------------------------------------------------
# File / ticker processing
# ---------------------------------------------------------------------------

def process_filing(path: Path, out_dir: Path) -> dict:
    data        = json.load(open(path))
    ticker      = data["ticker"]
    filing_id   = data["filing_id"]
    filing_type = data["filing_type"]
    filing_date = data["filing_date"]
    sections    = data.get("sections", {})

    if not sections:
        return {"filing_id": filing_id, "chunks": 0, "status": "empty"}

    all_chunks = []
    for sid, text in sections.items():
        chunks = semantic_chunk_section(
            text, ticker, filing_id, filing_type, filing_date, sid
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"filing_id": filing_id, "chunks": 0, "status": "no_chunks"}

    (out_dir / path.name).write_text(json.dumps({
        "filing_id"    : filing_id,
        "ticker"       : ticker,
        "filing_type"  : filing_type,
        "filing_date"  : filing_date,
        "total_chunks" : len(all_chunks),
        "chunks"       : all_chunks,
    }, indent=2, ensure_ascii=False))

    return {"filing_id": filing_id, "chunks": len(all_chunks), "status": "ok"}


def process_ticker(ticker: str) -> dict:
    in_dir  = PROCESSED_DIR / ticker
    out_dir = CHUNKS_DIR    / ticker

    if not in_dir.exists():
        log.warning(f"{ticker}: no processed directory")
        return {"ticker": ticker, "total_chunks": 0}

    files = sorted(in_dir.glob("*.json"))
    if not files:
        return {"ticker": ticker, "total_chunks": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in files:
        try:
            r = process_filing(f, out_dir)
            total += r["chunks"]
            log.info(f"  {f.name}: {r['chunks']} chunks")
        except Exception as e:
            log.error(f"  {f.name}: {e}")

    log.info(f"{ticker}: {total} total chunks")
    return {"ticker": ticker, "total_chunks": total}


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Chunking {len(tickers)} tickers...")
    total = 0
    results = []
    for t in tickers:
        r = process_ticker(t)
        results.append(r)
        total += r["total_chunks"]

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)
    print(f"  Total chunks: {total}")
    for r in results:
        if r["total_chunks"] > 0:
            print(f"    {r['ticker']:<8} {r['total_chunks']} chunks")


if __name__ == "__main__":
    main()