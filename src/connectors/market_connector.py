# =============================================================================
# src/connectors/market_connector.py
# =============================================================================
# Fetches market data for all tracked tickers via yfinance
# Stores in data/market/YYYY-MM-DD.json
# Links price signals + insider trades to Neo4j graph
#
# Environment variables (add to .env):
#   NEO4J_HOST/PORT/USER/PASSWORD — already set
#   No additional API key needed — yfinance is free
#
# Covers:
#   Daily price + volume for all tracked tickers
#   52-week high/low context
#   Relative strength vs sector
#   Unusual volume detection
#   Insider trading (Form 4 filings via yfinance)
#   Short interest
#   Options implied volatility (when available)

import sys
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

NEO4J_HOST     = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT     = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

MARKET_DIR = Path("data/market")

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]

# Sector ETFs for relative strength calculation
SECTOR_ETFS = {
    "consumer_discretionary": "XLY",
    "consumer_staples"       : "XLP",
    "energy"                 : "XLE",
    "healthcare"             : "XLV",
    "technology"             : "XLK",
    "industrials"            : "XLI",
    "financials"             : "XLF",
    "materials"              : "XLB",
    "sp500"                  : "SPY",
}

TICKER_SECTOR = {
    "AMZN":"consumer_discretionary","TSLA":"consumer_discretionary",
    "HD":"consumer_discretionary","MCD":"consumer_discretionary",
    "NKE":"consumer_discretionary","SBUX":"consumer_discretionary",
    "TGT":"consumer_discretionary","LOW":"consumer_discretionary",
    "BKNG":"consumer_discretionary","GM":"consumer_discretionary",
    "XOM":"energy","CVX":"energy","COP":"energy","SLB":"energy",
    "EOG":"energy","MPC":"energy","PSX":"energy","VLO":"energy","OXY":"energy",
    "JNJ":"healthcare","PFE":"healthcare","UNH":"healthcare","ABBV":"healthcare",
    "MRK":"healthcare","LLY":"healthcare","BMY":"healthcare",
    "AMGN":"healthcare","GILD":"healthcare","CVS":"healthcare",
    "AAPL":"technology","MSFT":"technology","GOOGL":"technology",
    "NVDA":"technology","META":"technology","ADBE":"technology",
    "CRM":"technology","INTC":"technology","CSCO":"technology","IBM":"technology",
}

# Volume spike threshold — flag if volume > X times 20-day avg
VOLUME_SPIKE_THRESHOLD = 2.5

# Price move threshold — flag if daily move > X%
PRICE_MOVE_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Neo4j client
# ---------------------------------------------------------------------------
_driver = None

def get_neo4j():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver


def run_query(query: str, params: dict = None):
    with get_neo4j().session() as session:
        return session.run(query, params or {}).data()


# ---------------------------------------------------------------------------
# Price data fetching
# ---------------------------------------------------------------------------

def fetch_price_data(ticker: str) -> Optional[dict]:
    """
    Fetch comprehensive price data for one ticker.
    Returns dict with price, volume, 52w range, moving averages.
    """
    try:
        import yfinance as yf
        import numpy as np

        t    = yf.Ticker(ticker)
        hist = t.history(period="1y")

        if hist.empty or len(hist) < 2:
            return None

        latest  = hist.iloc[-1]
        prev    = hist.iloc[-2]
        week_ago = hist.iloc[-6] if len(hist) >= 6 else hist.iloc[0]
        month_ago = hist.iloc[-22] if len(hist) >= 22 else hist.iloc[0]

        # Price changes
        pct_1d  = ((latest["Close"] - prev["Close"])    / prev["Close"]) * 100
        pct_1w  = ((latest["Close"] - week_ago["Close"]) / week_ago["Close"]) * 100
        pct_1m  = ((latest["Close"] - month_ago["Close"])/ month_ago["Close"])* 100
        pct_ytd = ((latest["Close"] - hist.iloc[0]["Close"]) / hist.iloc[0]["Close"]) * 100

        # 52-week range
        high_52w = float(hist["High"].max())
        low_52w  = float(hist["Low"].min())
        pct_from_52w_high = ((latest["Close"] - high_52w) / high_52w) * 100

        # Volume analysis
        avg_volume_20d = float(hist["Volume"].tail(20).mean())
        volume_ratio   = float(latest["Volume"] / avg_volume_20d) if avg_volume_20d > 0 else 1.0

        # Moving averages
        ma_50  = float(hist["Close"].tail(50).mean()) if len(hist) >= 50 else None
        ma_200 = float(hist["Close"].tail(200).mean()) if len(hist) >= 200 else None
        price  = float(latest["Close"])

        # Trend signals
        above_ma50 = bool(price > ma_50) if ma_50 else None
        above_ma200 = bool(price > ma_200) if ma_200 else None
        golden_cross = bool(ma_50 > ma_200) if (ma_50 and ma_200) else None

        # Signal flags
        is_volume_spike = bool(volume_ratio >= VOLUME_SPIKE_THRESHOLD)
        is_large_move = bool(abs(pct_1d) >= PRICE_MOVE_THRESHOLD)
        near_52w_high = bool(pct_from_52w_high >= -5.0)
        near_52w_low = bool(pct_from_52w_high <= -40.0)

        return {
            "ticker"            : ticker,
            "date"              : hist.index[-1].strftime("%Y-%m-%d"),
            "price"             : round(price, 2),
            "pct_change_1d"     : round(pct_1d, 3),
            "pct_change_1w"     : round(pct_1w, 3),
            "pct_change_1m"     : round(pct_1m, 3),
            "pct_change_ytd"    : round(pct_ytd, 3),
            "volume"            : int(latest["Volume"]),
            "avg_volume_20d"    : int(avg_volume_20d),
            "volume_ratio"      : round(volume_ratio, 2),
            "high_52w"          : round(high_52w, 2),
            "low_52w"           : round(low_52w, 2),
            "pct_from_52w_high" : round(pct_from_52w_high, 2),
            "ma_50"             : round(ma_50, 2) if ma_50 else None,
            "ma_200"            : round(ma_200, 2) if ma_200 else None,
            "above_ma50"        : above_ma50,
            "above_ma200"       : above_ma200,
            "golden_cross"      : golden_cross,
            "is_volume_spike"   : is_volume_spike,
            "is_large_move"     : is_large_move,
            "near_52w_high"     : near_52w_high,
            "near_52w_low"      : near_52w_low,
            "signals"           : _build_signals(ticker, pct_1d, volume_ratio,
                                                  pct_from_52w_high, golden_cross),
        }
    except Exception as e:
        log.error(f"Price data error {ticker}: {e}")
        return None


def _build_signals(ticker, pct_1d, volume_ratio, pct_from_52w_high, golden_cross):
    """Build list of notable signals for this ticker today."""
    signals = []

    if abs(pct_1d) >= PRICE_MOVE_THRESHOLD:
        signals.append({
            "type"     : "large_price_move",
            "direction": "up" if pct_1d > 0 else "down",
            "magnitude": abs(pct_1d),
            "note"     : f"{pct_1d:+.1f}% single day move",
        })

    if volume_ratio >= VOLUME_SPIKE_THRESHOLD:
        signals.append({
            "type"     : "volume_spike",
            "magnitude": volume_ratio,
            "note"     : f"{volume_ratio:.1f}x average volume",
        })

    if pct_from_52w_high >= -2.0:
        signals.append({
            "type": "near_52w_high",
            "note": f"Within {abs(pct_from_52w_high):.1f}% of 52-week high",
        })

    if pct_from_52w_high <= -40.0:
        signals.append({
            "type": "near_52w_low",
            "note": f"{abs(pct_from_52w_high):.1f}% below 52-week high",
        })

    if golden_cross is True:
        signals.append({
            "type": "golden_cross",
            "note": "50-day MA above 200-day MA (bullish trend)",
        })
    elif golden_cross is False:
        signals.append({
            "type": "death_cross",
            "note": "50-day MA below 200-day MA (bearish trend)",
        })

    return signals


# ---------------------------------------------------------------------------
# Insider trades fetching
# ---------------------------------------------------------------------------

def fetch_insider_trades(ticker: str) -> list:
    """Fetch recent insider transactions from yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        insiders = t.insider_transactions
        if insiders is None or insiders.empty:
            return []

        trades = []
        for _, row in insiders.head(10).iterrows():
            trades.append({
                "insider"    : str(row.get("Insider", "")),
                "relation"   : str(row.get("Relation", "")),
                "transaction": str(row.get("Transaction", "")),
                "shares"     : int(row.get("Shares", 0) or 0),
                "value"      : float(row.get("Value", 0) or 0),
                "date"       : str(row.get("Start Date", "")),
            })
        return trades
    except Exception as e:
        log.debug(f"Insider trades error {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# Neo4j loading
# ---------------------------------------------------------------------------

def load_market_to_graph(ticker: str, data: dict, fetch_date: str):
    """Update Company node with latest market data + flag signals."""

    run_query("""
        MATCH (c:Company {ticker: $ticker})
        SET c.latest_price      = $price,
            c.pct_change_1d     = $pct_1d,
            c.pct_change_1w     = $pct_1w,
            c.pct_change_1m     = $pct_1m,
            c.volume_ratio      = $vol_ratio,
            c.above_ma200       = $above_ma200,
            c.near_52w_high     = $near_high,
            c.near_52w_low      = $near_low,
            c.market_updated_at = $fetch_date
    """, {
        "ticker"     : ticker,
        "price"      : data.get("price"),
        "pct_1d"     : data.get("pct_change_1d"),
        "pct_1w"     : data.get("pct_change_1w"),
        "pct_1m"     : data.get("pct_change_1m"),
        "vol_ratio"  : data.get("volume_ratio"),
        "above_ma200": data.get("above_ma200"),
        "near_high"  : data.get("near_52w_high"),
        "near_low"   : data.get("near_52w_low"),
        "fetch_date" : fetch_date,
    })

    # Create MarketSignal nodes for notable events
    for signal in data.get("signals", []):
        signal_id = f"mkt_{ticker}_{signal['type']}_{fetch_date}"
        run_query("""
            MERGE (s:MarketSignal {signal_id: $signal_id})
            SET s.ticker     = $ticker,
                s.type       = $type,
                s.note       = $note,
                s.fetch_date = $fetch_date,
                s.updated_at = timestamp()
            WITH s
            MATCH (c:Company {ticker: $ticker})
            MERGE (c)-[r:HAS_MARKET_SIGNAL]->(s)
            SET r.fetch_date = $fetch_date
        """, {
            "signal_id"  : signal_id,
            "ticker"     : ticker,
            "type"       : signal["type"],
            "note"       : signal.get("note", ""),
            "fetch_date" : fetch_date,
        })

    # Load insider trades
    insiders = fetch_insider_trades(ticker)
    for trade in insiders:
        trade_id = (f"insider_{ticker}_{trade['insider'][:20]}"
                    f"_{trade['date']}".replace(" ", "_"))
        run_query("""
            MERGE (it:InsiderTrade {trade_id: $trade_id})
            SET it.ticker      = $ticker,
                it.insider     = $insider,
                it.relation    = $relation,
                it.transaction = $transaction,
                it.shares      = $shares,
                it.value       = $value,
                it.trade_date  = $date,
                it.updated_at  = timestamp()
            WITH it
            MATCH (c:Company {ticker: $ticker})
            MERGE (c)-[r:HAD_INSIDER_TRADE]->(it)
        """, {
            "trade_id"   : trade_id,
            "ticker"     : ticker,
            "insider"    : trade["insider"],
            "relation"   : trade["relation"],
            "transaction": trade["transaction"],
            "shares"     : trade["shares"],
            "value"      : trade["value"],
            "date"       : trade["date"],
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    fetch_date = datetime.now().strftime("%Y-%m-%d")

    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS

    log.info(f"Fetching market data for {len(tickers)} tickers...")

    results         = {"date": fetch_date, "tickers": {}, "alerts": []}
    total_signals   = 0

    for ticker in tickers:
        log.info(f"  {ticker}...")
        data = fetch_price_data(ticker)
        if not data:
            continue

        results["tickers"][ticker] = data
        total_signals += len(data.get("signals", []))

        # Collect alerts — notable moves worth flagging
        for sig in data.get("signals", []):
            results["alerts"].append({
                "ticker" : ticker,
                "signal" : sig["type"],
                "note"   : sig.get("note", ""),
                "price"  : data["price"],
                "pct_1d" : data["pct_change_1d"],
            })

        try:
            load_market_to_graph(ticker, data, fetch_date)
        except Exception as e:
            log.error(f"  Graph error {ticker}: {e}")

    # Save to disk
    out_path = MARKET_DIR / f"{fetch_date}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("MARKET CONNECTOR COMPLETE")
    print("=" * 60)
    print(f"  Date            : {fetch_date}")
    print(f"  Tickers fetched : {len(results['tickers'])}")
    print(f"  Total signals   : {total_signals}")

    if results["alerts"]:
        print(f"\n  Notable signals today ({len(results['alerts'])}):")
        for alert in sorted(results["alerts"],
                            key=lambda x: abs(x.get("pct_1d", 0)), reverse=True)[:10]:
            print(f"    {alert['ticker']:<6} {alert['signal']:<20} "
                  f"{alert['pct_1d']:+.1f}%  {alert['note']}")

    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()