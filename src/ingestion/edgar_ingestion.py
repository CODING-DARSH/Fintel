# =============================================================================
# src/ingestion/edgar_ingestion.py  —  Pull SEC filings from EDGAR
# =============================================================================
# What this does:
#   1. Resolves each ticker → CIK (company identifier in EDGAR)
#   2. Fetches filing metadata for 10-K and 8-K filings
#   3. Downloads the raw text of each filing
#   4. Saves to data/raw/filings/<TICKER>/<form_type>_<date>.json
#
# Why this matters (interview answer):
#   "We pulled directly from EDGAR's free API rather than a paid vendor.
#    This required handling rate limits, CIK resolution, and extracting
#    clean text from XBRL/HTML filing documents."

import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import config using sys.path so this works when run from any location
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_RAW, ALL_TICKERS, UNIVERSE,
    EDGAR_BASE, USER_AGENT, FILING_TYPES,
    START_DATE, END_DATE, LOGS
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOGS.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "edgar_ingestion.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": USER_AGENT}   # SEC requires this in every request


# ── CIK resolution ────────────────────────────────────────────────────────────

def get_cik(ticker: str) -> Optional[str]:
    """
    Resolve a ticker symbol to its SEC CIK number.
    EDGAR uses CIK as the primary company identifier.

    Returns zero-padded 10-digit CIK string, or None on failure.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry["ticker"].upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                log.info(f"  {ticker} → CIK {cik}")
                return cik

        log.warning(f"  CIK not found for {ticker}")
        return None

    except Exception as e:
        log.error(f"  CIK lookup failed for {ticker}: {e}")
        return None


# ── Filing metadata ────────────────────────────────────────────────────────────

def get_filings_metadata(cik: str, form_type: str) -> list[dict]:
    """
    Fetch list of all filings of a given type for this CIK.
    Returns list of dicts with: accessionNumber, filingDate, primaryDocument.

    Why we filter by date here:
      Avoids downloading filings outside our 3-year window —
      saves storage and prevents accidental look-ahead.
    """
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        submissions = resp.json()

        recent = submissions.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        dates       = recent.get("filingDate", [])
        accessions  = recent.get("accessionNumber", [])
        documents   = recent.get("primaryDocument", [])

        results = []
        for form, date, acc, doc in zip(forms, dates, accessions, documents):
            if form != form_type:
                continue
            if not (START_DATE <= date <= END_DATE):
                continue
            results.append({
                "form_type"       : form,
                "filing_date"     : date,
                "accession_number": acc.replace("-", ""),
                "primary_document": doc,
            })

        log.info(f"    Found {len(results)} {form_type} filings in date range")
        return results

    except Exception as e:
        log.error(f"    Metadata fetch failed: {e}")
        return []


# ── Filing text download ───────────────────────────────────────────────────────

def download_filing_text(cik: str, accession: str, document: str) -> Optional[str]:
    """
    Download the raw HTML/text of a single filing document.

    EDGAR filing URL structure:
      https://www.sec.gov/Archives/edgar/data/<CIK>/<accession>/<document>

    We return raw HTML here — cleaning happens in the cleaning module.
    This separation is deliberate: raw data should never be modified in place.
    """
    # Remove leading zeros from CIK for the URL
    cik_int = str(int(cik))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text

    except Exception as e:
        log.error(f"    Download failed ({url}): {e}")
        return None


# ── Save raw filing ────────────────────────────────────────────────────────────

def save_filing(ticker: str, metadata: dict, raw_text: str) -> Path:
    """
    Save raw filing + metadata as a single JSON.
    Schema:
      {
        "ticker":    "AAPL",
        "form_type": "10-K",
        "filing_date": "2023-11-03",
        "accession_number": "...",
        "raw_html": "...",
        "pulled_at": "2024-01-15T12:00:00",
        "source_url": "..."
      }

    Why JSON and not plain text:
      Keeps metadata co-located with content — easier to reload,
      trace provenance, and build a data card later.
    """
    out_dir = DATA_RAW / "filings" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{metadata['form_type'].replace('-','')}_{metadata['filing_date']}.json"
    out_path = out_dir / fname

    payload = {
        **metadata,
        "ticker"    : ticker,
        "raw_html"  : raw_text,
        "pulled_at" : datetime.utcnow().isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return out_path


# ── Main ingestion loop ────────────────────────────────────────────────────────

def ingest_ticker(ticker: str, form_types: list[str] = FILING_TYPES) -> dict:
    """
    Full ingestion pipeline for a single ticker.
    Returns a summary dict for logging/reporting.
    """
    log.info(f"Processing {ticker}")
    summary = {"ticker": ticker, "success": [], "failed": []}

    cik = get_cik(ticker)
    if not cik:
        summary["failed"].append("cik_resolution")
        return summary

    time.sleep(0.15)   # SEC rate limit: max ~10 req/sec; we stay well below

    for form_type in form_types:
        filings = get_filings_metadata(cik, form_type)

        for filing in filings:
            time.sleep(0.15)
            raw = download_filing_text(
                cik,
                filing["accession_number"],
                filing["primary_document"]
            )
            if raw:
                path = save_filing(ticker, filing, raw)
                summary["success"].append(str(path))
                log.info(f"    Saved → {path.name}")
            else:
                summary["failed"].append(filing["accession_number"])

    return summary


def run_ingestion(tickers: list[str] = None, dry_run: bool = False) -> list[dict]:
    """
    Run ingestion for all tickers (or a subset).

    dry_run=True: only resolves CIKs, does not download.
    Useful for testing your setup without hitting rate limits.
    """
    tickers = tickers or ALL_TICKERS
    all_summaries = []

    log.info(f"Starting EDGAR ingestion — {len(tickers)} tickers, "
             f"forms: {FILING_TYPES}, range: {START_DATE} → {END_DATE}")

    for i, ticker in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {ticker}")

        if dry_run:
            cik = get_cik(ticker)
            log.info(f"  DRY RUN — CIK={cik}, skipping download")
            all_summaries.append({"ticker": ticker, "cik": cik, "dry_run": True})
            time.sleep(0.1)
            continue

        summary = ingest_ticker(ticker)
        all_summaries.append(summary)

    # Save run manifest — important for reproducibility
    manifest_path = LOGS / f"edgar_manifest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "run_date"   : datetime.utcnow().isoformat(),
            "tickers"    : tickers,
            "start_date" : START_DATE,
            "end_date"   : END_DATE,
            "form_types" : FILING_TYPES,
            "summaries"  : all_summaries,
        }, f, indent=2)

    log.info(f"Manifest saved → {manifest_path}")
    return all_summaries


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start with 3 tickers as a smoke test before running all 50
    test_tickers = ["AAPL", "JPM", "XOM"]
    summaries = run_ingestion(tickers=test_tickers, dry_run=False)

    print("\n── Ingestion summary ──")
    for s in summaries:
        n_ok   = len(s.get("success", []))
        n_fail = len(s.get("failed", []))
        print(f"  {s['ticker']:6s}  saved={n_ok}  failed={n_fail}")