# =============================================================================
# src/cleaning/cleaning.py
# =============================================================================
# Uses BeautifulSoup for proper HTML parsing + ftfy for encoding fixes.
# Filters to financial sentences only — drops all legal/XBRL noise.

import re
import json
import logging
import ftfy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_RAW, DATA_PROCESSED, ALL_TICKERS, START_DATE, END_DATE, LOGS

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

# ── Noise patterns — drop any sentence matching these ─────────────────────────
_NOISE = re.compile(r"""
    # XBRL metadata
    [a-z]+-\d{8}\s+\d{7,}
    # Legal form headers
    | (securities\s+and\s+exchange\s+commission)
    | (check\s+the\s+appropriate\s+box)
    | (indicate\s+by\s+check\s+mark)
    | (emerging\s+growth\s+company)
    | (commission\s+file\s+number)
    | (state\s+or\s+other\s+jurisdiction)
    | (irs\s+employer)
    | (address\s+of\s+principal)
    | (registrant.s\s+telephone)
    | (trading\s+symbol)
    | (name\s+of\s+each\s+exchange)
    | (pursuant\s+to\s+rule\s+\d+)
    | (incorporated\s+(herein\s+)?by\s+reference)
    | (exhibit\s+\d+\.\d+)
    | (table\s+of\s+contents)
    | (form\s+(8-k|10-k|10-q|20-f))
    | (current\s+report\s+on\s+form)
    # Phone / zip
    | \(\d{3}\)\s*\d{3}[-\s]\d{4}
    | \b\d{5}(-\d{4})?\b.*zip
    # Pure numeric / code lines
    | ^[\d\s\-\/\.\,\(\)]+$
    # Checkbox artifacts
    | [☐☑✓]
""", re.IGNORECASE | re.VERBOSE)

# ── Financial keywords — sentence must have at least one ─────────────────────
_FINANCIAL = re.compile(
    r"revenue|income|profit|loss|margin|earning|growth|declin|increas|decreas"
    r"|quarter|fiscal|billion|million|percent|%|guidance|outlook|demand|supply"
    r"|cost|expense|cash|debt|equity|acqui|merger|divest|restructur"
    r"|headwind|tailwind|risk|opportunit|market|customer|product|service"
    r"|segment|operat|capital|invest|return|forecast|expect|perform"
    r"|result|sales|volume|pric|competit|regulat|litigat|settlement"
    r"|dividend|buyback|share|stock|interest\s+rate|inflation|recession"
    r"|workforce|employee|headcount|impairment|write|charge|goodwill",
    re.IGNORECASE
)

# ── Sentence splitter ─────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def extract_text(raw_html: str) -> str:
    """
    Parse HTML with BeautifulSoup, fix encoding with ftfy.
    Returns clean plain text.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script, style, and hidden elements
    for tag in soup(["script", "style", "meta", "link", "ix:header",
                     "ix:nonnumeric", "ix:nonfraction"]):
        tag.decompose()

    # Extract text from meaningful tags
    parts = []
    for tag in soup.find_all(["p", "div", "span", "td", "li", "h1",
                               "h2", "h3", "h4", "section"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    combined = " ".join(parts)
    combined = ftfy.fix_text(combined)           # fix â€œ → " etc.
    combined = re.sub(r'\s+', ' ', combined).strip()
    return combined


def is_financial(sentence: str) -> bool:
    """Keep only sentences with real financial content."""
    words = sentence.split()

    # Too short
    if len(words) < 8:
        return False

    # Too long (likely a data dump or table row)
    if len(words) > 120:
        return False

    # Matches noise pattern
    if _NOISE.search(sentence):
        return False

    # No financial keyword
    if not _FINANCIAL.search(sentence):
        return False

    # Mostly non-alphabetic (XBRL data, tables)
    alpha = sum(c.isalpha() for c in sentence)
    if alpha / max(len(sentence), 1) < 0.45:
        return False

    return True


def tokenise(text: str) -> list[str]:
    sentences = _SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if is_financial(s.strip())]


def clean_filing(raw_filing: dict) -> dict:
    raw_html = raw_filing.get("raw_html", "")

    text             = extract_text(raw_html)
    token_count_raw  = len(text.split())
    sentences        = tokenise(text)
    token_count_clean = sum(len(s.split()) for s in sentences)
    removed_pct      = round((1 - token_count_clean / max(token_count_raw, 1)) * 100, 1)

    return {
        "ticker"                  : raw_filing.get("ticker"),
        "form_type"               : raw_filing.get("form_type"),
        "filing_date"             : raw_filing.get("filing_date"),
        "accession_number"        : raw_filing.get("accession_number"),
        "sentences"               : sentences,
        "token_count_raw"         : token_count_raw,
        "token_count_clean"       : token_count_clean,
        "boilerplate_removed_pct" : removed_pct,
        "n_sentences"             : len(sentences),
    }


def run_filing_cleaning(tickers: list[str] = None) -> pd.DataFrame:
    tickers = tickers or ALL_TICKERS
    stats   = []

    for ticker in tickers:
        raw_dir = DATA_RAW / "filings" / ticker
        if not raw_dir.exists():
            log.warning(f"  {ticker}: no raw filings found")
            continue

        out_dir = DATA_PROCESSED / "filings" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for raw_path in sorted(raw_dir.glob("*.json")):
            with open(raw_path, encoding="utf-8") as f:
                raw = json.load(f)

            cleaned  = clean_filing(raw)
            out_path = out_dir / raw_path.name

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False)

            stats.append({
                "ticker"                 : ticker,
                "form_type"              : cleaned["form_type"],
                "filing_date"            : cleaned["filing_date"],
                "token_count_raw"        : cleaned["token_count_raw"],
                "token_count_clean"      : cleaned["token_count_clean"],
                "boilerplate_removed_pct": cleaned["boilerplate_removed_pct"],
                "n_sentences"            : cleaned["n_sentences"],
            })

            log.info(f"  {ticker} {cleaned['form_type']} {cleaned['filing_date']}: "
                     f"{cleaned['n_sentences']} financial sentences "
                     f"({cleaned['boilerplate_removed_pct']}% removed)")

    df = pd.DataFrame(stats)
    df.to_csv(DATA_PROCESSED / "filing_cleaning_stats.csv", index=False)
    return df


# ── Price cleaning (unchanged) ────────────────────────────────────────────────

def clean_prices(ticker: str) -> tuple[pd.DataFrame, dict]:
    raw_path = DATA_RAW / "prices" / f"{ticker}.csv"
    if not raw_path.exists():
        return pd.DataFrame(), {}

    df    = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df    = df.drop(columns=["ticker", "pulled_at"], errors="ignore").sort_index()
    df    = df[df.index.dayofweek < 5]
    stats = {"ticker": ticker, "original_rows": len(df)}

    full_idx     = pd.bdate_range(START_DATE, END_DATE)
    missing      = full_idx.difference(df.index)
    stats["missing_days_count"] = len(missing)
    stats["missing_days_pct"]   = round(len(missing) / len(full_idx) * 100, 2)

    df = df.reindex(full_idx).ffill(limit=3)

    if "Close" in df.columns:
        rets              = df["Close"].pct_change().dropna()
        z                 = (rets - rets.mean()) / rets.std()
        stats["outlier_count"] = int((z.abs() > 5).sum())
        stats["outlier_dates"] = list(rets[z.abs() > 5].index.astype(str))
        df["return_1d"]   = rets
        df["is_outlier"]  = (z.abs() > 5).reindex(df.index, fill_value=False)

    out_dir = DATA_PROCESSED / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{ticker}.csv")
    return df, stats


def run_price_cleaning(tickers: list[str] = None) -> pd.DataFrame:
    tickers   = tickers or ALL_TICKERS
    all_stats = []
    for ticker in tickers:
        _, stats = clean_prices(ticker)
        if stats:
            all_stats.append(stats)
    df = pd.DataFrame(all_stats)
    df.to_csv(DATA_PROCESSED / "price_cleaning_stats.csv", index=False)
    return df