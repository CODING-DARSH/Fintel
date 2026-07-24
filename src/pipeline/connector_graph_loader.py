# =============================================================================
# src/pipeline/connector_graph_loader.py
# =============================================================================
# Reads ALL accumulated connector data from disk and loads into Neo4j.
# Run this ONCE after graph_loader.py has already loaded filings.
# Then run daily after each connector run to keep graph fresh.
#
# Reads from:
#   data/news/extracted/YYYY-MM-DD/articles.json
#   data/macro/YYYY-MM-DD.json
#   data/market/YYYY-MM-DD.json
#
# Usage:
#   python src/pipeline/connector_graph_loader.py          # load everything
#   python src/pipeline/connector_graph_loader.py --today  # today only
#   python src/pipeline/connector_graph_loader.py --date 2024-01-15

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

NEWS_EXTRACTED_DIR = Path("data/news/extracted")
MACRO_DIR          = Path("data/macro")
MARKET_DIR         = Path("data/market")

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
        _driver.verify_connectivity()
        log.info("Neo4j connected.")
    return _driver


def run_query(query: str, params: dict = None):
    with get_neo4j().session() as session:
        return session.run(query, params or {}).data()


# ---------------------------------------------------------------------------
# Schema additions for connector nodes
# ---------------------------------------------------------------------------

def setup_connector_schema():
    constraints = [
        "CREATE CONSTRAINT news_article_id IF NOT EXISTS FOR (n:NewsArticle) REQUIRE n.article_id IS UNIQUE",
        "CREATE CONSTRAINT macro_signal_id IF NOT EXISTS FOR (m:MacroSignal) REQUIRE m.signal_id IS UNIQUE",
        "CREATE CONSTRAINT market_signal_id IF NOT EXISTS FOR (s:MarketSignal) REQUIRE s.signal_id IS UNIQUE",
        "CREATE CONSTRAINT insider_trade_id IF NOT EXISTS FOR (i:InsiderTrade) REQUIRE i.trade_id IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX news_date IF NOT EXISTS FOR (n:NewsArticle) ON (n.published_date)",
        "CREATE INDEX news_urgency IF NOT EXISTS FOR (n:NewsArticle) ON (n.urgency)",
        "CREATE INDEX macro_date IF NOT EXISTS FOR (m:MacroSignal) ON (m.fetch_date)",
        "CREATE INDEX macro_indicator IF NOT EXISTS FOR (m:MacroSignal) ON (m.indicator)",
        "CREATE INDEX market_date IF NOT EXISTS FOR (s:MarketSignal) ON (s.fetch_date)",
    ]
    for c in constraints:
        try:
            run_query(c)
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"Constraint: {e}")
    for i in indexes:
        try:
            run_query(i)
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"Index: {e}")
    log.info("Connector schema ready.")


# ---------------------------------------------------------------------------
# News loading
# ---------------------------------------------------------------------------

def load_news_file(filepath: Path) -> int:
    data     = json.load(open(filepath))
    loaded   = 0

    for article_data in data:
        article_id  = article_data.get("article_id")
        raw         = article_data.get("raw", {})
        extraction  = article_data.get("extraction", {})
        fetch_date  = article_data.get("fetch_date", "")

        if not article_id or not extraction:
            continue

        title      = raw.get("title", "")
        url        = raw.get("url", "")
        pub_date   = raw.get("publishedAt", fetch_date)[:10]
        event_type = extraction.get("event_type", "other")
        summary    = extraction.get("event_summary", title)
        urgency    = extraction.get("urgency", "medium")
        relevance  = extraction.get("relevance_score", 0.5)

        # Create NewsArticle node
        run_query("""
            MERGE (n:NewsArticle {article_id: $article_id})
            ON CREATE SET n.title          = $title,
                          n.url            = $url,
                          n.published_date = $pub_date,
                          n.event_type     = $event_type,
                          n.summary        = $summary,
                          n.urgency        = $urgency,
                          n.relevance      = $relevance,
                          n.created_at     = timestamp()
            ON MATCH SET  n.urgency        = $urgency,
                          n.relevance      = $relevance
        """, {
            "article_id": article_id, "title": title,
            "url": url, "pub_date": pub_date,
            "event_type": event_type, "summary": summary,
            "urgency": urgency, "relevance": relevance,
        })

        # Link to mentioned companies
        for company in extraction.get("companies_mentioned", []):
            ticker = (company.get("ticker") or "").upper()
            name   = company.get("name", "")
            role   = company.get("role", "mentioned")
            sent   = company.get("sentiment", "neutral")

            if not ticker and not name:
                continue

            node_id = ticker if ticker else name.upper().replace(" ", "_")[:15]

            run_query("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.name = $name, c.created_at = timestamp()
            """, {"ticker": node_id, "name": name or node_id})

            run_query("""
                MATCH (c:Company {ticker: $ticker})
                MATCH (n:NewsArticle {article_id: $article_id})
                MERGE (n)-[r:MENTIONS]->(c)
                SET r.role      = $role,
                    r.sentiment = $sent,
                    r.pub_date  = $pub_date
            """, {
                "ticker": node_id, "article_id": article_id,
                "role": role, "sent": sent, "pub_date": pub_date,
            })

        # Link supply chain signals
        for sc in extraction.get("supply_chain_signals", []):
            trigger = sc.get("trigger", "")
            geo     = sc.get("trigger_geography")
            inp     = sc.get("input_affected")
            lag     = sc.get("estimated_lag")

            if not trigger:
                continue

            event_id = f"news_{article_id}_{trigger[:30].lower().replace(' ','_')}"

            run_query("""
                MERGE (e:Event {event_id: $event_id})
                ON CREATE SET e.event_type     = 'news_trigger',
                              e.description    = $trigger,
                              e.geography      = $geo,
                              e.source_article = $article_id,
                              e.filing_date    = $pub_date,
                              e.created_at     = timestamp()
            """, {
                "event_id": event_id, "trigger": trigger,
                "geo": geo, "article_id": article_id, "pub_date": pub_date,
            })

            if inp:
                inp_id = inp.lower().replace(" ", "_")
                run_query("""
                    MERGE (i:Input {input_id: $inp_id})
                    ON CREATE SET i.name = $inp, i.created_at = timestamp()
                    WITH i
                    MATCH (e:Event {event_id: $event_id})
                    MERGE (e)-[r:DISRUPTS]->(i)
                    SET r.source_article = $article_id,
                        r.pub_date       = $pub_date
                """, {
                    "inp_id": inp_id, "inp": inp,
                    "event_id": event_id,
                    "article_id": article_id, "pub_date": pub_date,
                })

            for company in sc.get("companies_impacted", []):
                c_id = company.upper().replace(" ", "_")[:15]
                run_query("""
                    MERGE (c:Company {ticker: $ticker})
                    ON CREATE SET c.name = $name, c.created_at = timestamp()
                    WITH c
                    MATCH (e:Event {event_id: $event_id})
                    MERGE (e)-[r:PROPAGATES_TO]->(c)
                    SET r.source_article = $article_id,
                        r.lag_time       = $lag,
                        r.pub_date       = $pub_date
                """, {
                    "ticker": c_id, "name": company,
                    "event_id": event_id,
                    "article_id": article_id,
                    "lag": lag, "pub_date": pub_date,
                })

        # Link macro signals from news
        for macro in extraction.get("macro_signals", []):
            indicator = macro.get("indicator", "other")
            direction = macro.get("direction", "flat")
            geo       = macro.get("geography", "global")
            desc      = macro.get("description", "")
            signal_id = f"news_macro_{article_id}_{indicator}"

            run_query("""
                MERGE (m:MacroSignal {signal_id: $signal_id})
                ON CREATE SET m.indicator  = $indicator,
                              m.direction  = $direction,
                              m.geography  = $geo,
                              m.description= $desc,
                              m.source     = 'news',
                              m.fetch_date = $pub_date,
                              m.created_at = timestamp()
            """, {
                "signal_id": signal_id, "indicator": indicator,
                "direction": direction, "geo": geo,
                "desc": desc, "pub_date": pub_date,
            })

        loaded += 1

    return loaded


def load_all_news(date_filter: Optional[str] = None) -> int:
    if not NEWS_EXTRACTED_DIR.exists():
        log.warning("No news extracted directory found")
        return 0

    total = 0
    date_dirs = sorted(NEWS_EXTRACTED_DIR.iterdir())

    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue
        if date_filter and date_dir.name != date_filter:
            continue

        articles_file = date_dir / "articles.json"
        if not articles_file.exists():
            continue

        try:
            count = load_news_file(articles_file)
            total += count
            log.info(f"  News {date_dir.name}: {count} articles loaded")
        except Exception as e:
            log.error(f"  News {date_dir.name}: {e}")

    return total


# ---------------------------------------------------------------------------
# Macro loading
# ---------------------------------------------------------------------------

def load_macro_file(filepath: Path) -> int:
    data       = json.load(open(filepath))
    fetch_date = data.get("date", filepath.stem)
    loaded     = 0

    SECTOR_MAP = {
        "oil_wti"        : ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY",
                             "AMZN","GM","TSLA"],
        "oil_brent"      : ["XOM","CVX","COP","SLB","EOG"],
        "natural_gas"    : ["XOM","CVX","COP","SLB"],
        "copper"         : ["TSLA","GM"],
        "steel_etf"      : ["GM"],
        "fed_funds_rate" : ["ALL"],
        "treasury_10y"   : ["ALL"],
        "cpi_all_items"  : ["ALL"],
        "usd_index"      : ["AMZN","AAPL","MSFT","GOOGL","NVDA","MCD","SBUX","NKE"],
        "credit_spread_hy": ["ALL"],
    }

    # Load commodity + currency signals
    all_signals = {
        **data.get("commodities", {}),
        **data.get("currencies", {}),
    }

    for name, signal_data in all_signals.items():
        signal_id  = f"macro_{name}_{fetch_date}"
        pct        = signal_data.get("pct_change_1d", 0)
        direction  = signal_data.get("direction", "flat")
        magnitude  = signal_data.get("magnitude", "low")
        impact_dir = signal_data.get("impact_direction", "neutral")
        value      = signal_data.get("latest_close")

        run_query("""
            MERGE (m:MacroSignal {signal_id: $signal_id})
            SET m.indicator        = $name,
                m.value            = $value,
                m.pct_change_1d    = $pct,
                m.direction        = $direction,
                m.magnitude        = $magnitude,
                m.impact_direction = $impact,
                m.fetch_date       = $fetch_date,
                m.source           = 'yfinance',
                m.updated_at       = timestamp()
        """, {
            "signal_id": signal_id, "name": name,
            "value": str(value or ""), "pct": pct,
            "direction": direction, "magnitude": magnitude,
            "impact": impact_dir, "fetch_date": fetch_date,
        })

        # Link to affected companies
        affected = SECTOR_MAP.get(name, [])
        if "ALL" in affected:
            run_query("""
                MATCH (c:Company)
                MATCH (m:MacroSignal {signal_id: $signal_id})
                MERGE (m)-[r:AFFECTS]->(c)
                SET r.magnitude  = $magnitude,
                    r.fetch_date = $fetch_date
            """, {"signal_id": signal_id, "magnitude": magnitude,
                  "fetch_date": fetch_date})
        else:
            for ticker in affected:
                run_query("""
                    MATCH (c:Company {ticker: $ticker})
                    MATCH (m:MacroSignal {signal_id: $signal_id})
                    MERGE (m)-[r:AFFECTS]->(c)
                    SET r.magnitude  = $magnitude,
                        r.fetch_date = $fetch_date
                """, {"ticker": ticker, "signal_id": signal_id,
                      "magnitude": magnitude, "fetch_date": fetch_date})

        loaded += 1

    # Load FRED series latest values
    for name, observations in data.get("fred", {}).items():
        if not observations:
            continue
        latest    = observations[0]
        signal_id = f"fred_{name}_{fetch_date}"

        run_query("""
            MERGE (m:MacroSignal {signal_id: $signal_id})
            SET m.indicator  = $name,
                m.value      = $value,
                m.obs_date   = $obs_date,
                m.fetch_date = $fetch_date,
                m.source     = 'fred',
                m.updated_at = timestamp()
        """, {
            "signal_id": signal_id, "name": name,
            "value": latest.get("value", ""),
            "obs_date": latest.get("date", ""),
            "fetch_date": fetch_date,
        })
        loaded += 1

    return loaded


def load_all_macro(date_filter: Optional[str] = None) -> int:
    if not MACRO_DIR.exists():
        log.warning("No macro directory found")
        return 0

    total = 0
    for f in sorted(MACRO_DIR.glob("*.json")):
        date_str = f.stem
        if date_filter and date_str != date_filter:
            continue
        try:
            count = load_macro_file(f)
            total += count
            log.info(f"  Macro {date_str}: {count} signals loaded")
        except Exception as e:
            log.error(f"  Macro {date_str}: {e}")

    return total


# ---------------------------------------------------------------------------
# Market loading
# ---------------------------------------------------------------------------

def load_market_file(filepath: Path) -> int:
    data       = json.load(open(filepath))
    fetch_date = data.get("date", filepath.stem)
    loaded     = 0

    for ticker, tdata in data.get("tickers", {}).items():

        # Update Company node with latest market data
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
            "price"      : tdata.get("price"),
            "pct_1d"     : tdata.get("pct_change_1d"),
            "pct_1w"     : tdata.get("pct_change_1w"),
            "pct_1m"     : tdata.get("pct_change_1m"),
            "vol_ratio"  : tdata.get("volume_ratio"),
            "above_ma200": tdata.get("above_ma200"),
            "near_high"  : tdata.get("near_52w_high"),
            "near_low"   : tdata.get("near_52w_low"),
            "fetch_date" : fetch_date,
        })

        # Create MarketSignal nodes for notable events
        for signal in tdata.get("signals", []):
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
                "signal_id" : signal_id,
                "ticker"    : ticker,
                "type"      : signal["type"],
                "note"      : signal.get("note", ""),
                "fetch_date": fetch_date,
            })

        loaded += 1

    return loaded


def load_all_market(date_filter: Optional[str] = None) -> int:
    if not MARKET_DIR.exists():
        log.warning("No market directory found")
        return 0

    total = 0
    for f in sorted(MARKET_DIR.glob("*.json")):
        date_str = f.stem
        if date_filter and date_str != date_filter:
            continue
        try:
            count = load_market_file(f)
            total += count
            log.info(f"  Market {date_str}: {count} tickers loaded")
        except Exception as e:
            log.error(f"  Market {date_str}: {e}")

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = sys.argv[1:]
    today_only  = "--today" in args
    date_filter = None

    i = 0
    while i < len(args):
        if args[i] == "--date" and i + 1 < len(args):
            date_filter = args[i + 1]
            i += 2
        else:
            i += 1

    if today_only:
        date_filter = datetime.now().strftime("%Y-%m-%d")

    log.info("Setting up connector schema in Neo4j...")
    setup_connector_schema()

    log.info("Loading news data...")
    news_count = load_all_news(date_filter)

    log.info("Loading macro data...")
    macro_count = load_all_macro(date_filter)

    log.info("Loading market data...")
    market_count = load_all_market(date_filter)

    # Graph stats
    try:
        node_count = run_query("MATCH (n) RETURN count(n) as c")[0]["c"]
        rel_count  = run_query("MATCH ()-[r]->() RETURN count(r) as c")[0]["c"]
        news_nodes = run_query("MATCH (n:NewsArticle) RETURN count(n) as c")[0]["c"]
        macro_nodes= run_query("MATCH (n:MacroSignal) RETURN count(n) as c")[0]["c"]
        mkt_nodes  = run_query("MATCH (n:MarketSignal) RETURN count(n) as c")[0]["c"]
    except Exception:
        node_count = rel_count = news_nodes = macro_nodes = mkt_nodes = "?"

    print("\n" + "=" * 60)
    print("CONNECTOR GRAPH LOADER COMPLETE")
    print("=" * 60)
    if date_filter:
        print(f"  Date filter      : {date_filter}")
    print(f"  News loaded      : {news_count} articles")
    print(f"  Macro loaded     : {macro_count} signals")
    print(f"  Market loaded    : {market_count} tickers")
    print()
    print(f"  Graph totals:")
    print(f"    Total nodes    : {node_count}")
    print(f"    Total relations: {rel_count}")
    print(f"    NewsArticle    : {news_nodes}")
    print(f"    MacroSignal    : {macro_nodes}")
    print(f"    MarketSignal   : {mkt_nodes}")
    print()
    print(f"  Neo4j Browser   : http://localhost:7474")


if __name__ == "__main__":
    main()