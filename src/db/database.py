# =============================================================================
# src/db/database.py  —  Postgres connection + write helpers
# =============================================================================
# Single source of truth for all DB access in the pipeline.
# Every other module calls these functions — nothing connects to Postgres
# directly. This means if we ever change the DB, we change it here only.
#
# Why SQLAlchemy over raw psycopg2:
#   "We use SQLAlchemy so we can write DataFrames directly with .to_sql(),
#    handle connection pooling automatically, and swap the backend (e.g.
#    to SQLite for local testing) by changing one URL string."

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DB_URL

log = logging.getLogger(__name__)

# ── Engine (singleton) ────────────────────────────────────────────────────────
# pool_pre_ping=True: tests connection before using it —
# prevents "connection closed" errors after idle periods
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    return _engine


def db_available() -> bool:
    """
    Check if Postgres is reachable.
    Pipeline degrades gracefully if DB is unavailable —
    results still save to CSV, DB write is best-effort.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError:
        log.warning("Postgres not reachable — results will be saved to CSV only")
        return False


# ── Generic write helper ──────────────────────────────────────────────────────

def write_df(df: pd.DataFrame, table: str, if_exists: str = "append") -> bool:
    """
    Write a DataFrame to a Postgres table.

    if_exists options:
      'append'  — add rows (default, safe for incremental loads)
      'replace' — drop and recreate table (use only for full reloads)
      'fail'    — raise error if table exists

    Returns True on success, False on failure.
    """
    if df.empty:
        log.warning(f"  write_df: empty DataFrame, skipping {table}")
        return False

    if not db_available():
        return False

    try:
        df.to_sql(table, get_engine(), if_exists=if_exists, index=False, method="multi")
        log.info(f"  → {table}: wrote {len(df)} rows")
        return True
    except Exception as e:
        log.error(f"  write_df failed for {table}: {e}")
        return False


def read_df(query: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    if not db_available():
        return pd.DataFrame()
    try:
        return pd.read_sql(query, get_engine())
    except Exception as e:
        log.error(f"  read_df failed: {e}")
        return pd.DataFrame()


# ── Phase 1 specific writers ──────────────────────────────────────────────────

def write_labels(labels_df: pd.DataFrame) -> bool:
    """
    Write auto-labels to labels_auto table.
    Skips duplicates via ON CONFLICT (handled by if_exists='append'
    + unique constraint in schema).
    """
    cols = ["ticker", "form_type", "filing_date", "sentence_idx",
            "label_auto", "confidence"]
    score_cols = [c for c in labels_df.columns if c.startswith("score_")]
    keep = [c for c in cols + score_cols if c in labels_df.columns]

    df = labels_df[keep].copy()
    df = df.rename(columns={"label_auto": "label"})
    return write_df(df, "labels_auto")


def write_prices(ticker: str, price_df: pd.DataFrame) -> bool:
    """Write cleaned price data for one ticker."""
    df = price_df.copy().reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = ticker
    keep = ["ticker", "date", "open", "high", "low", "close",
            "volume", "return_1d", "is_outlier"]
    df = df[[c for c in keep if c in df.columns]]
    return write_df(df, "prices")


def write_forward_returns(returns_df: pd.DataFrame) -> bool:
    # Drop rows with any nulls before insert — schema has NOT NULL constraints
    df = returns_df.dropna(subset=["ticker","filing_date","forward_return","return_label","n_days"])
    dropped = len(returns_df) - len(df)
    if dropped:
        log.info(f"  write_forward_returns: dropped {dropped} null rows")
    return write_df(df, "forward_returns")


def write_macro(series_name: str, macro_df: pd.DataFrame) -> bool:
    df = macro_df.copy().reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["series_name"] = series_name
    df = df[["series_name", "date", "value"]]
    # VIX and some series have NaN on market holidays — drop before insert
    # Schema has NOT NULL constraint on value column
    before = len(df)
    df = df.dropna(subset=["value"])
    dropped = before - len(df)
    if dropped:
        log.info(f"  write_macro: dropped {dropped} null rows for {series_name} (holidays/weekends)")
    return write_df(df, "macro")


def write_eda_result(experiment: str, result: dict, phase: int = 1) -> bool:
    """Write a single EDA experiment result as a JSON row."""
    df = pd.DataFrame([{
        "experiment"  : experiment,
        "result_json" : json.dumps(result, default=str),
        "phase"       : phase,
    }])
    return write_df(df, "analysis_results")


def write_kappa(result: dict) -> bool:
    """Write Cohen's κ result to kappa_results table."""
    df = pd.DataFrame([{
        "kappa"           : result.get("kappa"),
        "observed_agree"  : result.get("observed_agree"),
        "n_samples"       : result.get("n_samples"),
        "gate_pass"       : result.get("gate_pass"),
        "per_class_json"  : json.dumps(result.get("per_class_agree", {})),
    }])
    return write_df(df, "annotation_agreement")


def write_data_quality(filing_stats: pd.DataFrame, price_stats: pd.DataFrame) -> bool:
    """Write combined data quality stats."""
    frames = []
    if not filing_stats.empty:
        frames.append(filing_stats)
    if not price_stats.empty:
        frames.append(price_stats)
    if not frames:
        return False
    combined = pd.concat(frames, ignore_index=True)
    return write_df(combined, "data_quality")


def write_splits(splits_df: pd.DataFrame) -> bool:
    return write_df(splits_df[["ticker", "filing_date", "split"]], "splits")