# =============================================================================
# src/connectors/macro_connector.py
# =============================================================================
# Fetches macro and commodity data from FRED + yfinance
# Stores in data/macro/YYYY-MM-DD.json
# Links signals to Neo4j graph
#
# Environment variables (add to .env):
#   FRED_API_KEY       — from fred.stlouisfed.org (free)
#   NEO4J_HOST/PORT/USER/PASSWORD — already set
#
# Covers:
#   Interest rates (Fed Funds, 10Y Treasury)
#   Inflation (CPI, PCE)
#   GDP growth
#   Unemployment
#   Oil prices (WTI, Brent)
#   Natural gas
#   Steel, copper, aluminum (commodity ETF proxies via yfinance)
#   USD index
#   Key currency pairs

import sys
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

FRED_API_KEY   = os.getenv("FRED_API_KEY", "")
NEO4J_HOST     = os.getenv("NEO4J_HOST", "localhost")
NEO4J_URI      = os.getenv("NEO4J_URI", "")
NEO4J_PORT     = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

MACRO_DIR = Path("data/macro")

# ---------------------------------------------------------------------------
# FRED series — economic indicators
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Interest rates
    "fed_funds_rate"       : "FEDFUNDS",
    "treasury_10y"         : "GS10",
    "treasury_2y"          : "GS2",
    "treasury_spread_10_2" : "T10Y2Y",
    # Inflation
    "cpi_all_items"        : "CPIAUCSL",
    "cpi_energy"           : "CPIENGSL",
    "pce_inflation"        : "PCEPI",
    "ppi_all"              : "PPIACO",
    # Growth
    "gdp_growth"           : "A191RL1Q225SBEA",
    "industrial_production": "INDPRO",
    "retail_sales"         : "RSAFS",
    # Labor
    "unemployment_rate"    : "UNRATE",
    "nonfarm_payrolls"     : "PAYEMS",
    # Credit
    "credit_spread_hy"     : "BAMLH0A0HYM2",
    "credit_spread_ig"     : "BAMLC0A0CM",
    # Housing
    "housing_starts"       : "HOUST",
}

# ---------------------------------------------------------------------------
# yfinance tickers — commodities + currencies
# ---------------------------------------------------------------------------
YFINANCE_TICKERS = {
    # Energy
    "oil_wti"          : "CL=F",
    "oil_brent"        : "BZ=F",
    "natural_gas"      : "NG=F",
    "gasoline"         : "RB=F",
    # Metals
    "gold"             : "GC=F",
    "silver"           : "SI=F",
    "copper"           : "HG=F",
    "aluminum_etf"     : "JJU",
    "steel_etf"        : "SLX",
    # Agriculture
    "corn"             : "ZC=F",
    "wheat"            : "ZW=F",
    "soybeans"         : "ZS=F",
    # Currencies
    "usd_index"        : "DX=F",
    "eur_usd"          : "EURUSD=X",
    "usd_jpy"          : "JPY=X",
    "usd_cny"          : "CNY=X",
    "usd_inr"          : "INR=X",
    # Shipping / logistics proxy
    "baltic_dry_etf"   : "BDRY",
}

# Which indicators directly affect which sectors
INDICATOR_SECTOR_MAP = {
    "oil_wti"          : ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY",
                          "AMZN","UPS","FDX","GM","TSLA"],
    "oil_brent"        : ["XOM","CVX","COP","SLB","EOG"],
    "natural_gas"      : ["XOM","CVX","COP","SLB"],
    "copper"           : ["TSLA","GM","GE","MMM"],
    "steel_etf"        : ["GM","F","CAT","DE","NUE"],
    "fed_funds_rate"   : ["ALL"],   # affects everyone
    "treasury_10y"     : ["ALL"],
    "cpi_all_items"    : ["ALL"],
    "usd_index"        : ["AMZN","AAPL","MSFT","GOOGL","NVDA","MCD","SBUX","NKE"],
    "credit_spread_hy" : ["ALL"],
}


# ---------------------------------------------------------------------------
# Neo4j client
# ---------------------------------------------------------------------------
_driver = None

def get_neo4j():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            NEO4J_URI if NEO4J_URI else f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver


def run_query(query: str, params: dict = None):
    with get_neo4j().session() as session:
        return session.run(query, params or {}).data()


# ---------------------------------------------------------------------------
# FRED data fetching
# ---------------------------------------------------------------------------

def fetch_fred(series_id: str, lookback_days: int = 30) -> list:
    """Fetch recent observations for a FRED series."""
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — skipping FRED data")
        return []

    import requests
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url   = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id"        : series_id,
        "api_key"          : FRED_API_KEY,
        "file_type"        : "json",
        "observation_start": start,
        "sort_order"       : "desc",
        "limit"            : 5,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        return [{"date": o["date"], "value": o["value"]} for o in obs
                if o["value"] != "."]
    except Exception as e:
        log.error(f"FRED error {series_id}: {e}")
        return []


# ---------------------------------------------------------------------------
# yfinance data fetching
# ---------------------------------------------------------------------------

def fetch_yfinance(ticker: str, period: str = "5d") -> dict:
    """Fetch recent price data for a commodity/currency ticker."""
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return {}

        latest    = hist.iloc[-1]
        prev      = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
        pct_chg   = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

        return {
            "ticker"        : ticker,
            "latest_close"  : round(float(latest["Close"]), 4),
            "prev_close"    : round(float(prev["Close"]), 4),
            "pct_change_1d" : round(float(pct_chg), 3),
            "volume"        : int(latest.get("Volume", 0)),
            "date"          : hist.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        log.error(f"yfinance error {ticker}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def classify_macro_signal(name: str, data: dict) -> dict:
    """
    Classify direction and magnitude of a macro move.
    Returns signal dict with direction, magnitude, affected_companies.
    """
    pct = data.get("pct_change_1d", 0)
    val = data.get("latest_close")

    if abs(pct) < 0.5:
        magnitude = "low"
    elif abs(pct) < 2.0:
        magnitude = "medium"
    elif abs(pct) < 5.0:
        magnitude = "high"
    else:
        magnitude = "critical"

    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"

    # Invert direction meaning for rates (rate up = bad for borrowers)
    cost_indicators = {"fed_funds_rate", "treasury_10y", "treasury_2y",
                       "credit_spread_hy", "credit_spread_ig"}
    if name in cost_indicators:
        impact_direction = "negative" if direction == "up" else "positive"
    else:
        impact_direction = "positive" if direction == "up" else "negative"

    affected = INDICATOR_SECTOR_MAP.get(name, [])

    return {
        "indicator"          : name,
        "value"              : val,
        "pct_change_1d"      : pct,
        "direction"          : direction,
        "magnitude"          : magnitude,
        "impact_direction"   : impact_direction,
        "affected_companies" : affected,
    }


# ---------------------------------------------------------------------------
# Neo4j loading
# ---------------------------------------------------------------------------

def load_macro_to_graph(name: str, signal: dict, fetch_date: str):
    """Create MacroSignal node and link to affected companies."""

    signal_id = f"macro_{name}_{fetch_date}"

    run_query("""
        MERGE (m:MacroSignal {signal_id: $signal_id})
        SET m.indicator = $indicator,
            m.value = $value,
            m.pct_change_1d = $pct,
            m.direction = $direction,
            m.magnitude = $magnitude,
            m.impact_direction = $impact,
            m.fetch_date = $fetch_date,
            m.updated_at = timestamp()
    """, {
        "signal_id" : signal_id,
        "indicator" : name,
        "value"     : str(signal.get("value", "")),
        "pct"       : signal.get("pct_change_1d", 0),
        "direction" : signal.get("direction", "flat"),
        "magnitude" : signal.get("magnitude", "low"),
        "impact"    : signal.get("impact_direction", "neutral"),
        "fetch_date": fetch_date,
    })

    # Link to affected companies
    affected = signal.get("affected_companies", [])
    if "ALL" in affected:
        # Link to all Company nodes
        run_query("""
            MATCH (c:Company)
            MATCH (m:MacroSignal {signal_id: $signal_id})
            MERGE (m)-[r:AFFECTS]->(c)
            SET r.magnitude = $magnitude,
                r.fetch_date = $fetch_date
        """, {
            "signal_id": signal_id,
            "magnitude": signal.get("magnitude", "low"),
            "fetch_date": fetch_date,
        })
    else:
        for ticker in affected:
            run_query("""
                MATCH (c:Company {ticker: $ticker})
                MATCH (m:MacroSignal {signal_id: $signal_id})
                MERGE (m)-[r:AFFECTS]->(c)
                SET r.magnitude = $magnitude,
                    r.fetch_date = $fetch_date
            """, {
                "ticker"    : ticker,
                "signal_id" : signal_id,
                "magnitude" : signal.get("magnitude", "low"),
                "fetch_date": fetch_date,
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    fetch_date = datetime.now().strftime("%Y-%m-%d")

    results = {"date": fetch_date, "fred": {}, "commodities": {},
               "currencies": {}, "signals": []}

    # Fetch FRED series
    log.info("Fetching FRED economic indicators...")
    for name, series_id in FRED_SERIES.items():
        obs = fetch_fred(series_id)
        if obs:
            results["fred"][name] = obs
            log.info(f"  {name}: {obs[0]['value']} ({obs[0]['date']})")

    # Fetch yfinance tickers
    log.info("Fetching commodity + currency prices...")
    for name, ticker in YFINANCE_TICKERS.items():
        data = fetch_yfinance(ticker)
        if data:
            signal = classify_macro_signal(name, data)
            results["commodities" if name not in
                    {"usd_index","eur_usd","usd_jpy","usd_cny","usd_inr"}
                    else "currencies"][name] = {**data, **signal}
            results["signals"].append(signal)

            if abs(data.get("pct_change_1d", 0)) >= 1.0:
                log.info(f"  ⚠ {name}: {data['pct_change_1d']:+.2f}% "
                         f"({signal['magnitude']} move)")

            try:
                load_macro_to_graph(name, signal, fetch_date)
            except Exception as e:
                log.error(f"  Graph error {name}: {e}")

    # Save to disk
    out_path = MACRO_DIR / f"{fetch_date}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary
    high_moves = [s for s in results["signals"]
                  if s["magnitude"] in ("high", "critical")]

    print("\n" + "=" * 60)
    print("MACRO CONNECTOR COMPLETE")
    print("=" * 60)
    print(f"  Date              : {fetch_date}")
    print(f"  FRED series       : {len(results['fred'])}")
    print(f"  Commodities       : {len(results['commodities'])}")
    print(f"  Currencies        : {len(results['currencies'])}")
    print(f"  High/critical moves: {len(high_moves)}")
    if high_moves:
        print("\n  Notable moves:")
        for s in high_moves:
            print(f"    {s['indicator']:<25} {s['pct_change_1d']:+.2f}% "
                  f"({s['magnitude']})")
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()