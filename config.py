# =============================================================================
# config.py — Central config for the entire Fintel platform
# =============================================================================
# All constants live here. No other file hard-codes keys, paths, or thresholds.
# Values load from .env first (local dev), then Docker-injected env vars.
#
# SECURITY: API keys are never written here directly.
#           They live in .env which is gitignored.

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_LABELS    = ROOT / "data" / "labels"
OUTPUTS        = ROOT / "outputs"
LOGS           = ROOT / "logs"

# ── Database ──────────────────────────────────────────────────────────────────
DB_USER     = os.getenv("POSTGRES_USER",     "fintel")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fintel_dev_password")
DB_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
DB_PORT     = os.getenv("POSTGRES_PORT",     "5432")
DB_NAME     = os.getenv("POSTGRES_DB",       "fintel")

# Neon (and most managed Postgres) requires SSL — local Docker Postgres
# doesn't need it and ignores the param harmlessly either way, so this
# is safe to always include rather than branching on host.
# channel_binding=require matches what Neon's own connection string
# includes by default; sslmode=require is the minimum SSL enforcement.
DB_SSLMODE  = os.getenv("POSTGRES_SSLMODE",  "require")
DB_CHANNEL_BINDING = os.getenv("POSTGRES_CHANNEL_BINDING", "")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"
if DB_CHANNEL_BINDING:
    DB_URL += f"&channel_binding={DB_CHANNEL_BINDING}"


# ── Groq (LLM) ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
# Swap to llama-3.1-70b-versatile for harder reasoning tasks
# Swap to mixtral-8x7b-32768 for long-context filing analysis

# ── Tavily (web search) ───────────────────────────────────────────────────────
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY",   "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# ── NewsAPI ───────────────────────────────────────────────────────────────────
NEWSAPI_KEY         = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_MAX_ARTICLES = int(os.getenv("NEWSAPI_MAX_ARTICLES", "10"))

# ── FRED ──────────────────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
USER_AGENT   = os.getenv("EDGAR_USER_AGENT", "FintelResearch admin@fintel-research.com")
EDGAR_BASE   = "https://data.sec.gov"
FILING_TYPES = ["10-K", "8-K"]

# ── ChromaDB (vector store) ───────────────────────────────────────────────────
CHROMA_HOST       = os.getenv("CHROMA_HOST",       "localhost")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT",   "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fintel_filings")

# ── Alert thresholds ──────────────────────────────────────────────────────────
ALERT_TIER1_THRESHOLD = float(os.getenv("ALERT_TIER1_THRESHOLD", "0.45"))
ALERT_TIER2_THRESHOLD = float(os.getenv("ALERT_TIER2_THRESHOLD", "0.75"))
ALERT_MIN_SIGNALS     = int(os.getenv("ALERT_MIN_SIGNALS",       "3"))

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE   = "2024-01-01"

# Train / val / test — strict chronological, never random
TRAIN_START = "2021-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-02-01"   # 20-day embargo after TRAIN_END
VAL_END     = "2023-06-30"
TEST_START  = "2023-08-01"   # 20-day embargo after VAL_END
TEST_END    = "2024-01-01"

# ── Stock universe ─────────────────────────────────────────────────────────────
UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META",
        "ADBE", "CRM",  "INTC",  "CSCO", "IBM",
    ],
    "Financials": [
        "JPM", "BAC", "GS",  "MS",  "WFC",
        "C",   "BLK", "AXP", "USB", "PNC",
    ],
    "Healthcare": [
        "JNJ", "PFE", "UNH", "ABBV", "MRK",
        "LLY", "BMY", "AMGN","GILD", "CVS",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "PXD", "MPC", "PSX", "VLO", "OXY",
    ],
    "ConsumerDiscretionary": [
        "AMZN", "TSLA", "HD",   "MCD",  "NKE",
        "SBUX", "TGT",  "LOW",  "BKNG", "GM",
    ],
}
ALL_TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]

# ── Knowledge graph — supply chain relationships ───────────────────────────────
# Format: company → list of companies it directly affects
# Depth 1 = direct. Agent traces depth 2 automatically via graph traversal.

# Relationship discovery config — no hardcoded relationships
RELATIONSHIP_MIN_FILING_MENTIONS = 3      # min times B appears in A's filing
RELATIONSHIP_MIN_PRICE_CORRELATION = 0.35 # min return correlation to be linked
RELATIONSHIP_MIN_NEWS_COMENTIONS = 5      # min news co-mentions in 90 days
RELATIONSHIP_DECAY_DAYS = 90              # relationships older than this get downweighted

# ── Scanner settings ──────────────────────────────────────────────────────────
SCANNER_INTERVAL_MINUTES  = int(os.getenv("SCANNER_INTERVAL_MINUTES",  "30"))
PRE_PREDICTION_DAYS_BEFORE = int(os.getenv("PRE_PREDICTION_DAYS_BEFORE","14"))

# ── Model / eval constants ────────────────────────────────────────────────────
RANDOM_SEED         = 42
KAPPA_GATE          = 0.70
CONFIDENCE_THRESH   = 0.85
FORWARD_RETURN_DAYS = 5
EMBARGO_DAYS        = 20

# ── Source trust hierarchy (used by ranker) ───────────────────────────────────
SOURCE_TRUST = {
    "sec_filing"    : 1.00,
    "company_ir"    : 0.95,
    "bloomberg"     : 0.90,
    "reuters"       : 0.90,
    "financial_times": 0.85,
    "wsj"           : 0.85,
    "analyst_report": 0.80,
    "newsapi"       : 0.65,
    "general_news"  : 0.60,
    "reddit"        : 0.30,
    "twitter"       : 0.25, 
    # Added for graph evidence sourced from connectors rather than SEC
    # filings — previously ALL graph evidence was silently scored as
    # sec_filing regardless of actual origin (see gather_evidence.py's
    # GRAPH_RESULT_TYPE_TRUST). Uncalibrated like the propagation
    # weights — reasonable priors, not measured. market_data is higher
    # than macro_data since it's the company's own real-time price/
    # volume data (less room for interpretation) vs. macro signals
    # which involve more indirect sector-mapping judgment calls.
    "market_data"   : 0.85,
    "macro_data"    : 0.75,
}

# ── Graph propagation edge weights (src/agents/analytics/graph_propagation.py) ─
# UNCALIBRATED PLACEHOLDER — these numbers are reasonable-looking guesses,
# not derived from any historical outcome data. Do not treat scores computed
# from these as predictive; they express "documented connection strength
# given available disclosures," nothing more. Revisit once propagation has
# been run against real cases where the actual outcome is known (e.g. did a
# supplier disruption measurably affect the dependent company), and the
# numbers can be checked against something real rather than asserted.
#
# Resolution priority for any given edge (see graph_propagation.py):
#   1. A real numeric field on the edge itself (percentage, cost_share)
#   2. This categorical mapping, keyed by whichever categorical field
#      the edge actually has (criticality / severity / dependency_level)
#   3. PROPAGATION_RELATION_DEFAULTS below, keyed by relationship type
PROPAGATION_CATEGORICAL_WEIGHTS = {
    "high"  : 0.85,
    "medium": 0.5,
    "low"   : 0.2,
}

# Fallback when an edge has neither a numeric field nor a categorical
# field populated at all. Also uncalibrated.
PROPAGATION_RELATION_DEFAULTS = {
    "DEPENDS_ON"    : 0.5,
    "SUPPLIES_TO"   : 0.5,
    "SOURCED_FROM"  : 0.6,
    "EXPOSED_TO"    : 0.5,
    "PROPAGATES_TO" : 0.6,
    "BUYS_FROM"     : 0.4,
}

# ── Embedding model (local, no API key needed) ────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_STRIDE    = 128