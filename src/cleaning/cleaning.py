# =============================================================================
# src/cleaning/cleaning.py  —  Data cleaning for filings + prices
# =============================================================================
# What this does:
#   1. Strips HTML boilerplate from SEC filings → clean sentence list
#   2. Handles price gaps, outlier detection, forward-fill
#   3. Records before/after stats for Experiment A and B (data quality report)
#
# Why this is a separate module (interview answer):
#   "We maintain a strict raw → processed separation. Raw files are
#    never modified. This means we can re-run cleaning with different
#    parameters without re-downloading data."

import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from html.parser import HTMLParser

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_RAW, DATA_PROCESSED, ALL_TICKERS,
    START_DATE, END_DATE, LOGS
)

LOGS.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "cleaning.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PART A:  Filing text cleaning
# ─────────────────────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper — removes tags, keeps text."""
    def __init__(self):
        super().__init__()
        self.reset()
        self._parts = []

    def handle_data(self, d):
        self._parts.append(d)

    def get_text(self) -> str:
        return " ".join(self._parts)


# Boilerplate patterns common in SEC filings
# Removing these reduces noise and improves NLP accuracy
_BOILERPLATE_PATTERNS = [
    r"table of contents",
    r"forward[- ]looking statements?",
    r"this (annual|quarterly) report on form (10-k|10-q|8-k)",
    r"(incorporated|filed) (herein )?by reference",
    r"see (accompanying )?notes to (consolidated )?financial statements",
    r"item \d+[a-z]?\.",
    r"part (i|ii|iii|iv)[^a-z]",
    r"^\s*page \d+\s*$",
    r"^\s*\d+\s*$",          # lone page numbers
    r"exhibit \d+\.\d+",
    r"pursuant to rule \d+[a-z]?-\d+",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS),
    re.IGNORECASE | re.MULTILINE
)

# Sentence tokenizer — split on period/exclamation/question followed by space + capital
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def strip_html(raw_html: str) -> str:
    """Remove HTML tags, decode entities, collapse whitespace."""
    stripper = _HTMLStripper()
    stripper.feed(raw_html)
    text = stripper.get_text()
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_boilerplate(text: str) -> tuple[str, float]:
    """
    Remove standard SEC filing boilerplate.
    Returns: (cleaned_text, fraction_removed)
    """
    original_len = len(text)
    # Replace boilerplate matches with a single space
    cleaned = _BOILERPLATE_RE.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    removed_frac = 1.0 - len(cleaned) / max(original_len, 1)
    return cleaned, removed_frac


def tokenise_sentences(text: str, min_words: int = 8) -> list[str]:
    """
    Split text into sentences.
    Filter out sentences shorter than min_words — these are usually
    headers, labels, or table entries, not useful for NLP.
    """
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if len(s.split()) >= min_words]


def clean_filing(raw_filing: dict) -> dict:
    """
    Full cleaning pipeline for one filing JSON.

    Input:  raw filing dict (from edgar_ingestion.py)
    Output: cleaned dict with:
              - sentences: list of clean sentences
              - token_count_raw: before cleaning
              - token_count_clean: after cleaning
              - boilerplate_removed_pct: % removed
              - n_sentences: number of usable sentences
    """
    raw_html = raw_filing.get("raw_html", "")

    # Step 1: strip HTML tags
    text = strip_html(raw_html)
    token_count_raw = len(text.split())

    # Step 2: remove boilerplate
    text, removed_frac = remove_boilerplate(text)

    # Step 3: split into sentences
    sentences = tokenise_sentences(text)
    token_count_clean = sum(len(s.split()) for s in sentences)

    return {
        "ticker"                   : raw_filing.get("ticker"),
        "form_type"                : raw_filing.get("form_type"),
        "filing_date"              : raw_filing.get("filing_date"),
        "accession_number"         : raw_filing.get("accession_number"),
        "sentences"                : sentences,
        "token_count_raw"          : token_count_raw,
        "token_count_clean"        : token_count_clean,
        "boilerplate_removed_pct"  : round(removed_frac * 100, 1),
        "n_sentences"              : len(sentences),
    }


def run_filing_cleaning(tickers: list[str] = None) -> pd.DataFrame:
    """
    Clean all raw filings for given tickers.
    Returns a DataFrame of per-filing stats (for Experiment A).
    """
    tickers = tickers or ALL_TICKERS
    stats   = []

    for ticker in tickers:
        raw_dir = DATA_RAW / "filings" / ticker
        if not raw_dir.exists():
            log.warning(f"  {ticker}: no raw filings found — run edgar_ingestion first")
            continue

        out_dir = DATA_PROCESSED / "filings" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for raw_path in sorted(raw_dir.glob("*.json")):
            with open(raw_path, encoding="utf-8") as f:
                raw = json.load(f)

            cleaned = clean_filing(raw)

            # Save processed filing
            out_path = out_dir / raw_path.name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False)

            stats.append({
                "ticker"                : ticker,
                "form_type"             : cleaned["form_type"],
                "filing_date"           : cleaned["filing_date"],
                "token_count_raw"       : cleaned["token_count_raw"],
                "token_count_clean"     : cleaned["token_count_clean"],
                "boilerplate_removed_pct": cleaned["boilerplate_removed_pct"],
                "n_sentences"           : cleaned["n_sentences"],
            })

            log.info(f"  {ticker} {cleaned['form_type']} {cleaned['filing_date']}: "
                     f"{cleaned['token_count_raw']}→{cleaned['token_count_clean']} tokens, "
                     f"{cleaned['boilerplate_removed_pct']}% boilerplate removed, "
                     f"{cleaned['n_sentences']} sentences")

    df = pd.DataFrame(stats)
    out_path = DATA_PROCESSED / "filing_cleaning_stats.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Filing cleaning stats saved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART B:  Price data cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_prices(ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Clean price data for one ticker.

    Steps:
      1. Load raw CSV
      2. Detect and flag missing trading days
      3. Forward-fill small gaps (≤3 days) — standard for price data
      4. Detect price outliers (returns > 5σ) — may indicate data errors
         vs. real events. We flag but do NOT drop — outliers may be real.
      5. Record stats for Experiment B

    Returns: (cleaned_df, stats_dict)
    """
    raw_path = DATA_RAW / "prices" / f"{ticker}.csv"
    if not raw_path.exists():
        log.warning(f"  {ticker}: no price file found")
        return pd.DataFrame(), {}

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df = df.drop(columns=["ticker", "pulled_at"], errors="ignore")
    df = df.sort_index()

    # Remove weekends just in case (shouldn't be there with yfinance)
    df = df[df.index.dayofweek < 5]

    stats = {"ticker": ticker, "original_rows": len(df)}

    # ── Missing days ──
    # Create full trading day index and find gaps
    full_idx      = pd.bdate_range(START_DATE, END_DATE)
    missing_days  = full_idx.difference(df.index)
    stats["missing_days_count"]   = len(missing_days)
    stats["missing_days_pct"]     = round(len(missing_days) / len(full_idx) * 100, 2)

    # ── Forward-fill gaps ≤3 consecutive days ──
    df = df.reindex(full_idx)
    df = df.ffill(limit=3)   # Holidays + occasional data gaps
    stats["rows_after_ffill"] = df.notna().sum()["Close"] if "Close" in df.columns else None

    # ── Outlier detection ──
    if "Close" in df.columns:
        returns = df["Close"].pct_change().dropna()
        zscore  = (returns - returns.mean()) / returns.std()
        outlier_mask = zscore.abs() > 5
        stats["outlier_count"] = int(outlier_mask.sum())
        stats["outlier_dates"] = list(returns[outlier_mask].index.astype(str))
        df["return_1d"]        = returns
        df["is_outlier"]       = outlier_mask.reindex(df.index, fill_value=False)

    # ── Save cleaned CSV ──
    out_dir  = DATA_PROCESSED / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"
    df.to_csv(out_path)

    log.info(f"  {ticker}: {stats['original_rows']} rows, "
             f"{stats['missing_days_count']} missing days "
             f"({stats['missing_days_pct']}%), "
             f"{stats.get('outlier_count', 0)} outliers flagged")
    return df, stats


def run_price_cleaning(tickers: list[str] = None) -> pd.DataFrame:
    """
    Clean all tickers. Returns DataFrame of stats (for Experiment B).
    """
    tickers   = tickers or ALL_TICKERS
    all_stats = []

    log.info(f"Cleaning prices for {len(tickers)} tickers")
    for ticker in tickers:
        _, stats = clean_prices(ticker)
        if stats:
            all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    out_path = DATA_PROCESSED / "price_cleaning_stats.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Price cleaning stats saved → {out_path}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickers = ["AAPL", "JPM", "XOM"]

    print("── Filing cleaning ──")
    filing_stats = run_filing_cleaning(tickers=test_tickers)
    if not filing_stats.empty:
        print(filing_stats[["ticker","form_type","token_count_raw",
                             "token_count_clean","boilerplate_removed_pct",
                             "n_sentences"]].to_string(index=False))
    else:
        print("  No filings found — run edgar_ingestion first")

    print("\n── Price cleaning ──")
    price_stats = run_price_cleaning(tickers=test_tickers)
    if not price_stats.empty:
        print(price_stats[["ticker","original_rows","missing_days_pct",
                           "outlier_count"]].to_string(index=False))
    else:
        print("  No price files found — run market_ingestion first")
