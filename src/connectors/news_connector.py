# =============================================================================
# src/connectors/news_connector.py
# =============================================================================
# Fetches financial news from NewsAPI and RSS feeds
# Stores raw articles in data/news/raw/YYYY-MM-DD/*.json
# Extracts entities + signals via shared Gemini+Groq rotation (see extract_article)
# Stores enriched articles in data/news/extracted/YYYY-MM-DD/*.json
# Links news events to existing Neo4j graph nodes
#
# Environment variables (add to .env):
#   NEWS_API_KEY       — from newsapi.org (free tier: 100 req/day)
#   GROQ_API_KEY       — already set
#   NEO4J_HOST         — already set
#   NEO4J_PORT         — already set
#   NEO4J_USER         — already set
#   NEO4J_PASSWORD     — already set
#
# Usage:
#   python src/connectors/news_connector.py           # fetch today
#   python src/connectors/news_connector.py --date 2024-01-15
#   python src/connectors/news_connector.py --tickers AAPL MSFT

import sys
import json
import logging
import os
import time
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Running this file directly (python src/connectors/news_connector.py)
# only adds THIS file's own directory (src/connectors) to sys.path, not
# the project root — needed here because extract_article() imports
# `from src.pipeline.extractor import call_llm`, a cross-package import
# that requires the project root to be on sys.path. Same fix as
# backtest_propagation.py, different parent depth since this file sits
# one level shallower under src/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
NEO4J_HOST     = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT     = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

NEWS_RAW_DIR       = Path("data/news/raw")
NEWS_EXTRACTED_DIR = Path("data/news/extracted")

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]

# Company name mapping for news search
TICKER_TO_NAME = {
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot",
    "MCD": "McDonald's", "NKE": "Nike", "SBUX": "Starbucks",
    "TGT": "Target", "LOW": "Lowe's", "BKNG": "Booking Holdings",
    "GM": "General Motors", "XOM": "ExxonMobil", "CVX": "Chevron",
    "COP": "ConocoPhillips", "SLB": "Schlumberger", "EOG": "EOG Resources",
    "MPC": "Marathon Petroleum", "PSX": "Phillips 66", "VLO": "Valero",
    "OXY": "Occidental Petroleum", "JNJ": "Johnson Johnson",
    "PFE": "Pfizer", "UNH": "UnitedHealth", "ABBV": "AbbVie",
    "MRK": "Merck", "LLY": "Eli Lilly", "BMY": "Bristol Myers",
    "AMGN": "Amgen", "GILD": "Gilead", "CVS": "CVS Health",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google Alphabet",
    "NVDA": "Nvidia", "META": "Meta Facebook", "ADBE": "Adobe",
    "CRM": "Salesforce", "INTC": "Intel", "CSCO": "Cisco", "IBM": "IBM",
}

# RSS feeds — free, no API key needed, always-on backup
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/companyNews",
    "https://www.ft.com/rss/home",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "https://feeds.bloomberg.com/markets/news.rss",
]

REQUESTS_PER_SECOND = 3
RETRY_DELAY         = 20

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
_groq   = None
_driver = None


def get_groq():
    """Kept only for backward compatibility — no longer used by
    extract_article(), which now uses the shared multi-provider
    call_llm from extractor.py instead of a Groq-only client. See
    extract_article()'s docstring for why this changed."""
    global _groq
    if _groq is None:
        from groq import Groq
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def get_neo4j():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        uri     = f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"
        _driver = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def run_query(query: str, params: dict = None):
    driver = get_neo4j()
    with driver.session() as session:
        return session.run(query, params or {}).data()


# ---------------------------------------------------------------------------
# NewsAPI fetching
# ---------------------------------------------------------------------------

def fetch_newsapi(query: str, from_date: str, page_size: int = 5) -> list:
    """Fetch articles from NewsAPI for a query string."""
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — skipping NewsAPI")
        return []

    import requests
    url = "https://newsapi.org/v2/everything"
    params = {
        "q"        : query,
        "from"     : from_date,
        "sortBy"   : "publishedAt",
        "language" : "en",
        "pageSize" : page_size,
        "apiKey"   : NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        log.error(f"NewsAPI error for '{query}': {e}")
        return []


def fetch_rss_feeds() -> list:
    """Fetch articles from RSS feeds — free, no API key needed."""
    import requests
    import xml.etree.ElementTree as ET

    articles = []
    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=10)
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            for item in items[:20]:
                title   = item.findtext("title", "")
                url     = item.findtext("link", "")
                desc    = item.findtext("description", "")
                pub     = item.findtext("pubDate", "")
                articles.append({
                    "title"      : title,
                    "url"        : url,
                    "description": desc,
                    "publishedAt": pub,
                    "source"     : {"name": feed_url.split("/")[2]},
                    "content"    : desc,
                })
        except Exception as e:
            log.warning(f"RSS feed error {feed_url}: {e}")

    return articles


def article_id(article: dict) -> str:
    """Stable unique ID for an article based on URL."""
    url = article.get("url", article.get("title", ""))
    return hashlib.md5(url.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Groq extraction for news articles
# ---------------------------------------------------------------------------

NEWS_EXTRACTION_PROMPT = """Extract structured financial intelligence from this news article.

Title: {title}
Source: {source}
Date: {date}
Content: {content}

Return ONLY a JSON object:
{{
  "relevance_score": 0.0,
  "companies_mentioned": [
    {{
      "ticker": "string or null",
      "name": "string",
      "role": "subject|mentioned|impacted|acquirer|target",
      "sentiment": "positive|negative|neutral"
    }}
  ],
  "event_type": "earnings|merger_acquisition|executive_change|regulatory|product_launch|partnership|supply_chain|macro|litigation|bankruptcy|restructuring|guidance|other",
  "event_summary": "string (one sentence)",
  "financial_impact": {{
    "metric": "string or null",
    "direction": "up|down|flat|unknown",
    "magnitude": "low|medium|high or null",
    "value": "string or null"
  }},
  "risk_signals": [
    {{
      "category": "string",
      "description": "string",
      "severity": "low|medium|high|critical",
      "affected_companies": ["ticker or name"]
    }}
  ],
  "supply_chain_signals": [
    {{
      "trigger": "string (what happened)",
      "trigger_geography": "string or null",
      "input_affected": "string or null",
      "companies_impacted": ["string"],
      "estimated_lag": "string or null"
    }}
  ],
  "macro_signals": [
    {{
      "indicator": "interest_rate|inflation|gdp|commodity_price|currency|trade_policy|other",
      "direction": "up|down|flat",
      "geography": "string",
      "description": "string"
    }}
  ],
  "forward_looking": true,
  "urgency": "low|medium|high|breaking",
  "key_entities": ["string"],
  "topics": ["string"]
}}"""


def extract_article(article: dict, attempt: int = 0) -> Optional[dict]:
    """
    Extract structured signals from one news article.

    CHANGED: previously called Groq directly and exclusively — meaning
    when Groq's daily quota ran out, news extraction had NO fallback at
    all, even though Gemini keys (used by extractor.py) might still have
    budget. Now uses the shared multi-provider call_llm from
    extractor.py, same rotation across Gemini + Groq keys the filing
    extraction pipeline already uses. This fixes the immediate "Groq
    quota exhausted, news extraction fully blocked" problem, and also
    consolidates what used to be two separate, duplicated LLM-calling
    implementations into one.
    """
    from src.pipeline.extractor import call_llm

    content = (article.get("content") or article.get("description") or "")[:2000]
    prompt  = NEWS_EXTRACTION_PROMPT.format(
        title   = article.get("title", ""),
        source  = article.get("source", {}).get("name", ""),
        date    = article.get("publishedAt", ""),
        content = content,
    )
    return call_llm(prompt)


# ---------------------------------------------------------------------------
# Neo4j — link news events to existing graph
# ---------------------------------------------------------------------------

def load_news_to_graph(article: dict, extraction: dict, article_id: str):
    """Link extracted news signals into the existing Neo4j graph."""
    if not extraction:
        return

    pub_date    = article.get("publishedAt", "")[:10]
    title       = article.get("title", "")
    url         = article.get("url", "")
    event_type  = extraction.get("event_type", "other")
    summary     = extraction.get("event_summary", title)
    urgency     = extraction.get("urgency", "medium")

    # Create NewsArticle node
    run_query("""
        MERGE (n:NewsArticle {article_id: $article_id})
        ON CREATE SET n.title = $title, n.url = $url,
                      n.published_date = $pub_date,
                      n.event_type = $event_type,
                      n.summary = $summary,
                      n.urgency = $urgency,
                      n.created_at = timestamp()
    """, {
        "article_id": article_id, "title": title, "url": url,
        "pub_date": pub_date, "event_type": event_type,
        "summary": summary, "urgency": urgency,
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
            SET r.role = $role, r.sentiment = $sent,
                r.pub_date = $pub_date
        """, {
            "ticker": node_id, "article_id": article_id,
            "role": role, "sent": sent, "pub_date": pub_date,
        })

    # Link supply chain signals to graph
    for sc in extraction.get("supply_chain_signals", []):
        trigger  = sc.get("trigger", "")
        geo      = sc.get("trigger_geography")
        inp      = sc.get("input_affected")

        if not trigger:
            continue

        event_node_id = f"news_{article_id}_{trigger[:30].lower().replace(' ','_')}"
        run_query("""
            MERGE (e:Event {event_id: $event_id})
            ON CREATE SET e.event_type = 'news_trigger',
                          e.description = $trigger,
                          e.geography = $geo,
                          e.source_article = $article_id,
                          e.filing_date = $pub_date,
                          e.created_at = timestamp()
        """, {
            "event_id": event_node_id, "trigger": trigger,
            "geo": geo, "article_id": article_id, "pub_date": pub_date,
        })

        # Link event to affected input in graph
        if inp:
            inp_id = inp.lower().replace(" ", "_")
            run_query("""
                MERGE (i:Input {input_id: $inp_id})
                ON CREATE SET i.name = $inp, i.created_at = timestamp()
                WITH i
                MATCH (e:Event {event_id: $event_id})
                MERGE (e)-[r:DISRUPTS]->(i)
                SET r.source_article = $article_id,
                    r.pub_date = $pub_date
            """, {
                "inp_id": inp_id, "inp": inp,
                "event_id": event_node_id,
                "article_id": article_id, "pub_date": pub_date,
            })

        # Link event to impacted companies
        for company in sc.get("companies_impacted", []):
            c_id = company.upper().replace(" ", "_")[:15]
            run_query("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.name = $name, c.created_at = timestamp()
                WITH c
                MATCH (e:Event {event_id: $event_id})
                MERGE (e)-[r:PROPAGATES_TO]->(c)
                SET r.source_article = $article_id,
                    r.lag_time = $lag,
                    r.pub_date = $pub_date
            """, {
                "ticker": c_id, "name": company,
                "event_id": event_node_id,
                "article_id": article_id,
                "lag": sc.get("estimated_lag"),
                "pub_date": pub_date,
            })


# ---------------------------------------------------------------------------
# Main fetch + extract + load loop
# ---------------------------------------------------------------------------

def run(tickers: list, fetch_date: str):
    NEWS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    date_raw_dir = NEWS_RAW_DIR       / fetch_date
    date_ext_dir = NEWS_EXTRACTED_DIR / fetch_date
    date_raw_dir.mkdir(exist_ok=True)
    date_ext_dir.mkdir(exist_ok=True)

    all_articles = []

    # Fetch from NewsAPI for each ticker
    if NEWS_API_KEY:
        log.info(f"Fetching from NewsAPI for {len(tickers)} tickers...")
        for ticker in tickers:
            name  = TICKER_TO_NAME.get(ticker, ticker)
            query = f"{name} OR {ticker}"
            arts  = fetch_newsapi(query, fetch_date, page_size=3)
            for a in arts:
                a["_ticker_query"] = ticker
            all_articles.extend(arts)
            time.sleep(0.5)  # NewsAPI rate limit
        log.info(f"  NewsAPI: {len(all_articles)} articles fetched")

    # Always fetch RSS as supplement
    log.info("Fetching RSS feeds...")
    rss_articles = fetch_rss_feeds()
    all_articles.extend(rss_articles)
    log.info(f"  RSS: {len(rss_articles)} articles fetched")

    # Deduplicate by article_id
    seen    = set()
    unique  = []
    for a in all_articles:
        aid = article_id(a)
        if aid not in seen:
            seen.add(aid)
            unique.append((aid, a))
    log.info(f"Total unique articles: {len(unique)}")

    # Save raw
    raw_path = date_raw_dir / "articles.json"
    raw_path.write_text(json.dumps(
        [a for _, a in unique], indent=2, ensure_ascii=False
    ))

    # Extract + load to graph — written to disk after EACH article, not
    # once at the end. A batched single write-at-the-end (the original
    # approach) means a crash/kill mid-run (Ctrl+C, rate-limit
    # exhaustion, container timeout — all things that have actually
    # happened today) loses every successfully-extracted article that
    # hadn't been written yet, even though load_news_to_graph() already
    # wrote them into Neo4j in real time. Per-article save closes that
    # gap: the graph and the local JSON record stay in sync, and a
    # future run can tell what's already been processed instead of
    # starting blind. Matches extractor.py's save_chunk() pattern.
    ext_path = date_ext_dir / "articles.json"
    extracted = []

    def _save_extracted():
        ext_path.write_text(json.dumps(extracted, indent=2, ensure_ascii=False))

    for i, (aid, article) in enumerate(unique):
        log.info(f"  [{i+1}/{len(unique)}] Extracting: {article.get('title','')[:60]}")
        ext = extract_article(article)
        enriched = {
            "article_id" : aid,
            "raw"        : article,
            "extraction" : ext or {},
            "fetch_date" : fetch_date,
        }
        extracted.append(enriched)
        _save_extracted()   # crash-safe: write after EVERY article, not just at the end

        if ext:
            try:
                load_news_to_graph(article, ext, aid)
            except Exception as e:
                log.error(f"  Graph load error: {e}")

        time.sleep(1.0 / REQUESTS_PER_SECOND)

    log.info(f"Done. Raw: {raw_path} | Extracted: {ext_path}")
    return len(extracted)


def main():
    args     = sys.argv[1:]
    tickers  = []
    date_str = datetime.now().strftime("%Y-%m-%d")

    i = 0
    while i < len(args):
        if args[i] == "--date" and i + 1 < len(args):
            date_str = args[i + 1]
            i += 2
        elif args[i] == "--tickers" and i + 1 < len(args):
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                tickers.append(args[i].upper())
                i += 1
        else:
            i += 1

    if not tickers:
        tickers = TICKERS

    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — will use RSS feeds only")
    # NOTE: extract_article() now uses the shared call_llm from
    # extractor.py, which expects GEMINI_API_KEY_1/2/3 and/or
    # GROQ_API_KEY_1/2/3 (not the old single GROQ_API_KEY this used to
    # check). extractor.py itself validates key presence on import and
    # exits with a clear message if none are found — no separate check
    # needed here, and the old check was stale/wrong after this file
    # stopped using GROQ_API_KEY directly.

    log.info(f"Fetching news for {len(tickers)} tickers on {date_str}")
    count = run(tickers, date_str)

    print("\n" + "=" * 60)
    print("NEWS CONNECTOR COMPLETE")
    print("=" * 60)
    print(f"  Articles processed : {count}")
    print(f"  Date               : {date_str}")
    print(f"  Raw output         : {NEWS_RAW_DIR / date_str}")
    print(f"  Extracted output   : {NEWS_EXTRACTED_DIR / date_str}")


if __name__ == "__main__":
    main()  