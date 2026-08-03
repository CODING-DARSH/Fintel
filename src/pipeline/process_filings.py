# =============================================================================
# src/pipeline/process_filings.py
# =============================================================================
# Reads raw JSON filings, runs them through the appropriate parser,
# saves cleaned sections to data/processed/<TICKER>/<TYPE>_<DATE>.json
#
# Supports:
#   10-K  → section_splitter.py
#   8-K   → section_splitter_8k.py
#   (future: DEF14A, S-1, merger docs, Indian filings etc — add elif below)
#
# Usage:
#   python src/pipeline/process_filings.py            # process all tickers
#   python src/pipeline/process_filings.py AAPL MSFT  # specific tickers only

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.extraction.section_splitter    import split_filing
from src.extraction.section_splitter_8k import split_8k

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw/filings")
PROCESSED_DIR = Path("data/processed")

TICKERS = [
    "AMZN", "TSLA", "HD",   "MCD",  "NKE",
    "SBUX", "TGT",  "LOW",  "BKNG", "GM",
    "XOM",  "CVX",  "COP",  "SLB",  "EOG",
    "PXD",  "MPC",  "PSX",  "VLO",  "OXY",
    "JNJ",  "PFE",  "UNH",  "ABBV", "MRK",
    "LLY",  "BMY",  "AMGN", "GILD", "CVS",
    "AAPL", "MSFT", "GOOGL","NVDA", "META",
    "ADBE", "CRM",  "INTC", "CSCO", "IBM",
]


def process_file(raw_path: Path, out_dir: Path) -> dict:
    """
    Parse one raw filing file and save cleaned output.
    Returns a summary dict for reporting.
    """
    raw = json.load(open(raw_path))
    filename    = raw_path.name          # e.g. 10K_2022-10-28.json
    filing_date = raw.get("filing_date", "unknown")
    ticker      = raw.get("ticker", out_dir.name)

    # Determine filing type from filename prefix
    if filename.startswith("10K_"):
        filing_type = "10-K"
        result      = split_filing(raw["raw_html"], ticker=ticker, filing_date=filing_date)
        sections    = result["sections"]
        status      = result["confidence"]["overall_status"]

    elif filename.startswith("8K_"):
        filing_type = "8-K"
        result      = split_8k(raw["raw_html"], ticker=ticker, filing_date=filing_date)
        sections    = result["sections"]
        status      = result["confidence"]["overall_status"]

    else:
        # Unknown type — skip for now, placeholder for future types
        # Future: "DEF14A_", "S1_", "merger_", "IN_" (Indian filings) etc.
        log.warning(f"Unknown filing type: {filename} — skipping")
        return {"status": "skipped", "reason": "unknown_type"}

    # Nothing extracted — still save so we know this file was processed
    if not sections:
        output = {
            "ticker"      : ticker,
            "filing_type" : filing_type,
            "filing_date" : filing_date,
            "filing_id"   : f"{ticker}_{filing_type.replace('-', '')}_{filing_date}",
            "status"      : status,
            "sections"    : {},
        }
    else:
        output = {
            "ticker"      : ticker,
            "filing_type" : filing_type,
            "filing_date" : filing_date,
            "filing_id"   : f"{ticker}_{filing_type.replace('-', '')}_{filing_date}",
            "status"      : status,
            "sections"    : sections,
        }

    # Save to data/processed/TICKER/TYPE_DATE.json
    out_path = out_dir / filename
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    return {
        "status"        : status,
        "filing_type"   : filing_type,
        "sections_count": len(sections),
        "out_path"      : str(out_path),
    }


def process_ticker(ticker: str) -> dict:
    raw_dir = RAW_DIR / ticker
    out_dir = PROCESSED_DIR / ticker

    if not raw_dir.exists():
        log.warning(f"{ticker}: no raw directory found")
        return {"ticker": ticker, "status": "no_raw_dir"}

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        log.warning(f"{ticker}: no raw files found")
        return {"ticker": ticker, "status": "no_files"}

    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"10-K": 0, "8-K": 0, "skipped": 0, "failed": 0}

    for raw_path in raw_files:
        try:
            summary = process_file(raw_path, out_dir)
            ft = summary.get("filing_type", "skipped")
            if ft in counts:
                counts[ft] += 1
            elif summary.get("status") == "skipped":
                counts["skipped"] += 1
        except Exception as e:
            log.error(f"{ticker}/{raw_path.name}: {e}")
            counts["failed"] += 1

    log.info(
        f"{ticker}: 10-K={counts['10-K']} 8-K={counts['8-K']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    return {"ticker": ticker, **counts}


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Processing {len(tickers)} tickers...")
    results = []
    for ticker in tickers:
        results.append(process_ticker(ticker))

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    total_10k = sum(r.get("10-K", 0) for r in results)
    total_8k  = sum(r.get("8-K",  0) for r in results)
    failed    = sum(r.get("failed", 0) for r in results)
    print(f"  10-K files processed : {total_10k}")
    print(f"  8-K  files processed : {total_8k}")
    print(f"  Failed               : {failed}")
    print(f"  Output directory     : {PROCESSED_DIR.resolve()}")

    # Show what was saved
    print("\nProcessed structure:")
    for ticker_dir in sorted(PROCESSED_DIR.iterdir()):
        files = list(ticker_dir.glob("*.json"))
        tenk  = [f for f in files if f.name.startswith("10K_")]
        eightk= [f for f in files if f.name.startswith("8K_")]
        print(f"  {ticker_dir.name:<8} {len(tenk)} 10-K  {len(eightk)} 8-K")


if __name__ == "__main__":
    main()