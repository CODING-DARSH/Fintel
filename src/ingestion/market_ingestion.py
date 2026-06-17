# =============================================================================
# src/ingestion/market_ingestion.py  —  Pull price + macro data
# =============================================================================
# What this does:
#   1. Downloads OHLCV price data via yfinance for all tickers
#   2. Adjusts for splits and dividends (critical for return calculation)
#   3. Downloads macro series from FRED
#   4. Saves clean CSVs to data/raw/prices/ and data/raw/macro/
#
# Why adjusted prices matter (interview answer):
#   "If you don't adjust for splits, a 2:1 stock split looks like a
#    50% price drop — your model learns a ghost signal. We used
#    yfinance auto-adjust to handle this."

import json
import requests
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_RAW, ALL_TICKERS, START_DATE, END_DATE,
    FRED_API_KEY, FRED_SERIES, LOGS, RANDOM_SEED
)

LOGS.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "market_ingestion.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ── Price ingestion ───────────────────────────────────────────────────────────


def _fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV directly from Yahoo Finance v8 chart API.
    Bypasses yfinance timezone lookup which fails in some Docker environments.
    Returns adjusted-close DataFrame indexed by date.
    """
    import time as _time
    import calendar

    # Convert dates to Unix timestamps (Yahoo API requirement)
    def to_ts(date_str):
        import datetime as dt
        d = dt.datetime.strptime(date_str, "%Y-%m-%d")
        return int(calendar.timegm(d.timetuple()))

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={to_ts(start)}&period2={to_ts(end)}"
        f"&interval=1d&events=div,splits&includeAdjustedClose=true"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            result = data["chart"]["result"]
            if not result:
                return pd.DataFrame()

            r         = result[0]
            timestamps = r["timestamp"]
            quote      = r["indicators"]["quote"][0]
            adjclose   = r["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])

            df = pd.DataFrame({
                "Open"   : quote["open"],
                "High"   : quote["high"],
                "Low"    : quote["low"],
                "Close"  : adjclose,          # adjusted close
                "Volume" : quote["volume"],
            }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").normalize())

            df.index = df.index.tz_localize(None)   # strip tz for Postgres compatibility
            df.index.name = "Date"
            df = df.dropna(subset=["Close"])
            return df

        except Exception as e:
            log.warning(f"  {ticker} attempt {attempt+1}/3 failed: {e}")
            _time.sleep(2)

    return pd.DataFrame()


def pull_prices(tickers: list[str] = None) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for all tickers.

    auto_adjust=True:  prices adjusted for splits + dividends
    Why: ensures returns are economically meaningful

    Returns dict: {ticker → DataFrame with columns [Open, High, Low, Close, Volume]}
    Also saves per-ticker CSV to data/raw/prices/
    """
    tickers = tickers or ALL_TICKERS
    out_dir = DATA_RAW / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed  = []

    log.info(f"Pulling prices for {len(tickers)} tickers: {START_DATE} → {END_DATE}")

    for ticker in tickers:
        try:
            df = _fetch_yahoo(ticker, START_DATE, END_DATE)

            if df.empty:
                log.warning(f"  {ticker}: empty response")
                failed.append(ticker)
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Add metadata columns
            df["ticker"]     = ticker
            df["pulled_at"]  = datetime.utcnow().isoformat()

            # Save raw CSV
            out_path = out_dir / f"{ticker}.csv"
            df.to_csv(out_path)
            results[ticker] = df

            log.info(f"  {ticker}: {len(df)} trading days → {out_path.name}")

        except Exception as e:
            log.error(f"  {ticker}: failed — {e}")
            failed.append(ticker)

    # Save pull manifest
    manifest = {
        "pulled_at"  : datetime.utcnow().isoformat(),
        "start_date" : START_DATE,
        "end_date"   : END_DATE,
        "n_success"  : len(results),
        "n_failed"   : len(failed),
        "failed"     : failed,
    }
    with open(LOGS / "prices_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Price pull complete. Success: {len(results)}, Failed: {len(failed)}")
    if failed:
        log.warning(f"Failed tickers: {failed}")

    return results


# ── Macro ingestion ───────────────────────────────────────────────────────────

def pull_macro() -> dict[str, pd.DataFrame]:
    """
    Pull macroeconomic series from FRED.

    Why include macro (interview answer):
      "Macro variables like yield curve slope and VIX are known
       systematic risk factors. Including them lets the model
       distinguish company-specific sentiment from macro-driven
       sentiment — an important separation for institutional users."

    Returns dict: {series_name → DataFrame}
    """
    out_dir = DATA_RAW / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try importing fredapi — gracefully degrade if not installed
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        use_fred = (FRED_API_KEY != "YOUR_FRED_API_KEY_HERE")
    except ImportError:
        log.warning("fredapi not installed — run: pip install fredapi")
        use_fred = False

    results = {}

    if not use_fred:
        log.warning("FRED API key not configured — using synthetic macro data for development")
        # Generate synthetic data so rest of pipeline still runs
        idx = pd.date_range(START_DATE, END_DATE, freq="MS")
        for name in FRED_SERIES:
            df = pd.DataFrame({"value": 1.0}, index=idx)
            df.index.name = "date"
            df["series"] = name
            out_path = out_dir / f"{name}.csv"
            df.to_csv(out_path)
            results[name] = df
            log.info(f"  {name}: synthetic placeholder saved")
        return results

    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=END_DATE,
            )
            df = s.to_frame(name="value")
            df.index.name = "date"
            df["series"] = name

            out_path = out_dir / f"{name}.csv"
            df.to_csv(out_path)
            results[name] = df

            log.info(f"  {name} ({series_id}): {len(df)} observations → {out_path.name}")

        except Exception as e:
            log.error(f"  {name} ({series_id}): failed — {e}")

    return results


# ── Load helpers (used by other modules) ─────────────────────────────────────

def load_prices(ticker: str) -> Optional[pd.DataFrame]:
    """Load previously saved price CSV for one ticker."""
    path = DATA_RAW / "prices" / f"{ticker}.csv"
    if not path.exists():
        log.warning(f"Price file not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Remove metadata columns before returning
    return df.drop(columns=["ticker", "pulled_at"], errors="ignore")


def load_all_prices() -> pd.DataFrame:
    """
    Load all tickers into a single wide DataFrame (multi-index columns).
    Columns: (OHLCV_field, ticker)
    """
    frames = {}
    for ticker in ALL_TICKERS:
        df = load_prices(ticker)
        if df is not None:
            frames[ticker] = df

    if not frames:
        raise FileNotFoundError("No price files found. Run pull_prices() first.")

    combined = pd.concat(frames, axis=1)
    log.info(f"Loaded prices: {len(frames)} tickers, {len(combined)} trading days")
    return combined


def load_macro(series_name: str) -> Optional[pd.DataFrame]:
    """Load a single macro series CSV."""
    path = DATA_RAW / "macro" / f"{series_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run smoke test with 5 tickers
    test_tickers = ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]

    print("── Pulling prices ──")
    price_data = pull_prices(tickers=test_tickers)
    for t, df in price_data.items():
        print(f"  {t:6s}: {len(df)} rows, cols={list(df.columns[:5])}")

    print("\n── Pulling macro ──")
    macro_data = pull_macro()
    for name, df in macro_data.items():
        print(f"  {name:15s}: {len(df)} observations")

    print("\nDone. Check data/raw/prices/ and data/raw/macro/")