# =============================================================================
# src/retrieval/query_parser.py
# =============================================================================
# Parses natural language queries into structured retrieval parameters.
# Decides which retrievers to call and with what filters.
#
# INTENT TYPES:
#   company_profile    — "tell me about AAPL"
#   risk_query         — "what risks does AMZN face"
#   shared_risk        — "which companies face fuel cost risk"
#   supply_chain       — "what does TSLA depend on"
#   propagation        — "how does mine disruption affect GM"
#   competitor_query   — "who competes with MSFT in cloud"
#   causal_query       — "what caused margin compression at XOM"
#   macro_impact       — "which companies are affected by oil prices"
#   news_query         — "recent news about NVDA"
#   executive_query    — "CEO changes in pharma"
#   geographic_query   — "companies exposed to China"
#   market_signal      — "unusual volume or price moves today"
#   general            — fallback to vector search

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# Known tickers — extend as needed
KNOWN_TICKERS = {
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
    "TATASTEEL","MARUTI","RELIANCE","INFY","TCS","WIPRO","HDFCBANK",
    "UPS","FDX","WMT","COST","TGT","AMGN","BRK","JPM","BAC","GS",
}

# Ticker aliases — company name → ticker
COMPANY_ALIASES = {
    "amazon": "AMZN", "tesla": "TSLA", "apple": "AAPL",
    "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "pfizer": "PFE", "johnson": "JNJ", "merck": "MRK",
    "walmart": "WMT", "home depot": "HD", "target": "TGT",
    "starbucks": "SBUX", "mcdonalds": "MCD", "nike": "NKE",
    "intel": "INTC", "cisco": "CSCO", "ibm": "IBM",
    "abbvie": "ABBV", "gilead": "GILD", "amgen": "AMGN",
    "booking": "BKNG", "general motors": "GM",
    "tata steel": "TATASTEEL", "maruti": "MARUTI",
}

# Risk keywords → risk category
RISK_KEYWORDS = {
    "supply chain"     : "supply_chain_risk",
    "fuel cost"        : "macro_risk",
    "fuel"             : "macro_risk",
    "oil price"        : "macro_risk",
    "energy cost"      : "macro_risk",
    "inflation"        : "macro_risk",
    "interest rate"    : "macro_risk",
    "currency"         : "macro_risk",
    "regulatory"       : "regulatory_risk",
    "regulation"       : "regulatory_risk",
    "litigation"       : "litigation_risk",
    "lawsuit"          : "litigation_risk",
    "cyber"            : "cyber_risk",
    "competition"      : "competition_risk",
    "competitive"      : "competition_risk",
    "execution"        : "execution_risk",
    "talent"           : "talent_risk",
    "esg"              : "esg_risk",
    "geopolitical"     : "geopolitical_risk",
    "technology"       : "technology_risk",
}

# Macro indicator keywords
MACRO_KEYWORDS = {
    "oil"      : "oil_wti",
    "crude"    : "oil_wti",
    "gas"      : "natural_gas",
    "steel"    : "steel_etf",
    "copper"   : "copper",
    "gold"     : "gold",
    "fed rate" : "fed_funds_rate",
    "interest" : "fed_funds_rate",
    "inflation": "cpi_all_items",
    "cpi"      : "cpi_all_items",
    "dollar"   : "usd_index",
    "usd"      : "usd_index",
    "yuan"     : "usd_cny",
    "euro"     : "eur_usd",
}

# Section relevance by intent
INTENT_SECTIONS = {
    "risk_query"     : ["1A"],
    "shared_risk"    : ["1A"],
    "supply_chain"   : ["1A", "1", "2"],
    "propagation"    : ["1A", "1"],
    "causal_query"   : ["7", "1A"],
    "company_profile": ["1", "7"],
    "competitor_query": ["1", "1A"],
    "macro_impact"   : ["7", "7A", "1A"],
    "news_query"     : [],   # news nodes, not chunks
    "executive_query": [],   # person nodes
    "market_signal"  : [],   # market signal nodes
    "general"        : ["1A", "7"],
}


@dataclass
class ParsedQuery:
    """Structured retrieval parameters extracted from natural language."""
    original_query   : str
    intent           : str   = "general"
    tickers          : list  = field(default_factory=list)
    risk_categories  : list  = field(default_factory=list)
    macro_indicators : list  = field(default_factory=list)
    geographies      : list  = field(default_factory=list)
    sections         : list  = field(default_factory=list)
    filing_types     : list  = field(default_factory=list)
    date_from        : Optional[str] = None
    date_to          : Optional[str] = None
    use_graph        : bool  = True
    use_vector       : bool  = True
    graph_query_type : Optional[str] = None
    keywords         : list  = field(default_factory=list)
    top_k            : int   = 10

    def to_dict(self) -> dict:
        return self.__dict__


class QueryParser:
    """
    Parses natural language queries into structured retrieval params.
    Rule-based — no LLM call needed, fast, deterministic.
    For complex queries the agent layer can override these defaults.
    """

    def parse(self, query: str, top_k: int = 10) -> ParsedQuery:
        """
        Parse a natural language query into retrieval parameters.

        Examples:
            "What risks does Apple face?" →
                intent=risk_query, tickers=[AAPL], sections=[1A]

            "Which companies are exposed to fuel cost risk?" →
                intent=shared_risk, risk_categories=[macro_risk],
                keywords=[fuel cost]

            "How would a mine disruption affect Tesla's supply chain?" →
                intent=propagation, tickers=[TSLA]
        """
        q_lower  = query.lower().strip()
        pq       = ParsedQuery(original_query=query, top_k=top_k)

        # 1. Extract tickers and company names
        pq.tickers = self._extract_tickers(query, q_lower)

        # 2. Detect intent
        pq.intent = self._detect_intent(q_lower)

        # 3. Extract risk categories
        pq.risk_categories = self._extract_risks(q_lower)

        # 4. Extract macro indicators
        pq.macro_indicators = self._extract_macro(q_lower)

        # 5. Extract geographies
        pq.geographies = self._extract_geographies(q_lower)

        # 6. Extract date filters
        pq.date_from, pq.date_to = self._extract_dates(q_lower)

        # 7. Extract filing type preference
        if any(x in q_lower for x in ["annual", "10-k", "10k", "yearly"]):
            pq.filing_types = ["10-K"]
        elif any(x in q_lower for x in ["8-k", "8k", "earnings", "event"]):
            pq.filing_types = ["8-K"]

        # 8. Set sections based on intent
        pq.sections = INTENT_SECTIONS.get(pq.intent, ["1A", "7"])

        # 9. Set graph query type
        pq.graph_query_type = self._map_intent_to_graph(pq.intent)

        # 10. Decide which retrievers to use
        pq.use_graph  = pq.intent not in {"general"}
        pq.use_vector = True  # always use vector

        # 11. Extract keywords for BM25
        pq.keywords = self._extract_keywords(q_lower)

        log.info(f"Parsed query: intent={pq.intent} tickers={pq.tickers} "
                 f"risks={pq.risk_categories} sections={pq.sections}")
        return pq

    def _extract_tickers(self, query: str, q_lower: str) -> list:
        """Extract ticker symbols and company names from query."""
        found = []

        # Direct ticker match (uppercase 2-6 chars)
        for token in re.findall(r"\b[A-Z]{2,6}\b", query):
            if token in KNOWN_TICKERS:
                found.append(token)

        # Company name aliases
        for alias, ticker in COMPANY_ALIASES.items():
            if alias in q_lower and ticker not in found:
                found.append(ticker)

        return list(dict.fromkeys(found))  # deduplicate, preserve order

    def _detect_intent(self, q: str) -> str:
        """Rule-based intent detection from query text."""
        # Order matters — more specific patterns first

        if any(x in q for x in ["propagat", "chain reaction", "downstream",
                                  "affect if", "impact if", "what if",
                                  "mine disrupt", "port congest"]):
            return "propagation"

        if any(x in q for x in ["supply chain", "depend on", "depends on",
                                  "input", "supplier", "raw material",
                                  "sourced from"]):
            return "supply_chain"

        if any(x in q for x in ["who else", "which companies", "other companies",
                                  "all companies", "sector", "industry"]):
            if any(x in q for x in ["risk", "exposed", "face", "affect"]):
                return "shared_risk"

        if any(x in q for x in ["risk", "exposed to", "faces", "vulnerable",
                                  "threat", "concern"]):
            if len(re.findall(r"\b[A-Z]{2,6}\b", q)) > 0:
                return "risk_query"
            return "shared_risk"

        if any(x in q for x in ["compet", "rival", "vs", "versus",
                                  "compare", "benchmark"]):
            return "competitor_query"

        if any(x in q for x in ["ceo", "cfo", "executive", "officer",
                                  "departure", "appoint", "resign", "hired"]):
            return "executive_query"

        if any(x in q for x in ["news", "recent", "latest", "today",
                                  "announced", "filed", "reported"]):
            return "news_query"

        if any(x in q for x in ["oil", "steel", "copper", "fed rate",
                                  "inflation", "macro", "commodity",
                                  "interest rate", "currency"]):
            return "macro_impact"

        if any(x in q for x in ["caused", "why did", "reason for",
                                  "led to", "resulted in"]):
            return "causal_query"

        if any(x in q for x in ["china", "india", "europe", "asia",
                                  "geography", "region", "country"]):
            return "geographic_query"

        if any(x in q for x in ["volume", "price move", "52 week",
                                  "insider", "short interest", "options"]):
            return "market_signal"

        if any(x in q for x in ["tell me about", "overview", "profile",
                                  "summary of"]):
            return "company_profile"

        return "general"

    def _extract_risks(self, q: str) -> list:
        found = []
        for keyword, category in RISK_KEYWORDS.items():
            if keyword in q and category not in found:
                found.append(category)
        return found

    def _extract_macro(self, q: str) -> list:
        found = []
        for keyword, indicator in MACRO_KEYWORDS.items():
            if keyword in q and indicator not in found:
                found.append(indicator)
        return found

    def _extract_geographies(self, q: str) -> list:
        # Each canonical geography maps to itself plus common demonym/
        # adjective forms — plain substring matching missed "Chinese"
        # (does not contain "china" as a substring: chin-A vs chin-ESE),
        # so a real query about "the Chinese export ban" silently found
        # no geography at all. Word-boundary matching (via regex \b)
        # also avoids the opposite failure mode — a bare substring check
        # would match "china" inside an unrelated word.
        geo_forms = {
            "china"        : ["china", "chinese"],
            "india"        : ["india", "indian"],
            "europe"       : ["europe", "european"],
            "usa"          : ["usa", "u.s.a", "u.s."],
            "united states": ["united states", "american"],
            "japan"        : ["japan", "japanese"],
            "australia"    : ["australia", "australian"],
            "brazil"       : ["brazil", "brazilian"],
            "russia"       : ["russia", "russian"],
            "middle east"  : ["middle east", "middle eastern"],
            "asia"         : ["asia", "asian"],
            "latin america": ["latin america", "latin american"],
            "africa"       : ["africa", "african"],
            "canada"       : ["canada", "canadian"],
            "mexico"       : ["mexico", "mexican"],
            "germany"      : ["germany", "german"],
            "uk"           : ["uk", "u.k."],
            "united kingdom": ["united kingdom", "british"],
            "france"       : ["france", "french"],
            "south korea"  : ["south korea", "korean", "south korean"],
            "taiwan"       : ["taiwan", "taiwanese"],
        }
        found = []
        for canonical, forms in geo_forms.items():
            for form in forms:
                if re.search(r"\b" + re.escape(form) + r"\b", q):
                    found.append(canonical)
                    break
        return found

    def _extract_dates(self, q: str) -> tuple:
        """Extract year-based date filters."""
        date_from = date_to = None
        years = re.findall(r"\b(20\d{2})\b", q)
        if len(years) >= 2:
            date_from = f"{min(years)}-01-01"
            date_to   = f"{max(years)}-12-31"
        elif len(years) == 1:
            date_from = f"{years[0]}-01-01"
            date_to   = f"{years[0]}-12-31"
        if "last year" in q or "past year" in q:
            from datetime import datetime, timedelta
            now = datetime.now()
            date_from = (now - timedelta(days=365)).strftime("%Y-%m-%d")
            date_to   = now.strftime("%Y-%m-%d")
        return date_from, date_to

    def _extract_keywords(self, q: str) -> list:
        """Extract meaningful keywords for BM25."""
        stops = {
            "what","which","how","who","when","where","why","is","are",
            "was","were","does","do","did","has","have","had","will","would",
            "could","should","the","a","an","and","or","but","in","on","at",
            "to","for","of","with","by","from","tell","me","about","give",
        }
        words = re.findall(r"\b[a-z][a-z]{2,}\b", q)
        return [w for w in words if w not in stops]

    def _map_intent_to_graph(self, intent: str) -> Optional[str]:
        return {
            "risk_query"      : "get_company_risks",
            "shared_risk"     : "get_shared_risk_companies",
            "supply_chain"    : "get_supply_chain",
            "propagation"     : "get_propagation_risks",
            "competitor_query": "get_competitors",
            "causal_query"    : "get_causal_chains",
            "macro_impact"    : "get_macro_impact",
            "news_query"      : "get_news_impact",
            "executive_query" : "get_executive_changes",
            "geographic_query": "get_geographic_exposure",
            "market_signal"   : "get_market_signals",
            "company_profile" : "get_company_overview",
        }.get(intent)