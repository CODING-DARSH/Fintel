# =============================================================================
# src/retrieval/test_retrieval.py
# =============================================================================
# Production-grade evaluation of the retrieval layer.
#
# METRICS:
#   Recall@K    — did at least one relevant result appear in top K?
#   Precision@K — fraction of top K results that are relevant
#   MRR         — Mean Reciprocal Rank of first relevant result
#   NDCG@K      — Normalized Discounted Cumulative Gain
#   Hit Rate    — binary: did query return ANY result
#   Latency     — response time in ms
#   Coverage    — how many query types work end to end
#
# TEST CATEGORIES:
#   1. Company risk queries     (single company, known risk)
#   2. Cross-company queries    (shared risk, sector-wide)
#   3. Supply chain queries     (dependency traversal)
#   4. Propagation queries      (multi-hop event → company)
#   5. Competitor queries       (competitive landscape)
#   6. Macro impact queries     (commodity/rate → company)
#   7. News queries             (recent events)
#   8. Executive queries        (people changes)
#   9. Causal chain queries     (cause → effect)
#   10. Geographic queries      (country/region exposure)
#
# Usage:
#   python src/retrieval/test_retrieval.py                    # full eval
#   python src/retrieval/test_retrieval.py --quick            # 5 queries only
#   python src/retrieval/test_retrieval.py --category risk    # one category
#   python src/retrieval/test_retrieval.py --graph-only       # graph only
#   python src/retrieval/test_retrieval.py --vector-only      # vector only

import sys
import json
import time
import logging
import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.graph_retriever  import GraphRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.retrieval_merger import RetrievalMerger
from src.retrieval.query_parser     import QueryParser

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# TEST CASES
# Each test case defines:
#   query           — natural language query
#   category        — test category
#   expected_tickers— tickers that MUST appear in results (for Recall@K)
#   expected_types  — result types that should appear
#   relevant_keywords—keywords that should appear in result text
#   min_results     — minimum number of results expected
#   k               — K for Recall@K / Precision@K (default 5)
#   use_graph       — whether to test graph retrieval
#   use_vector      — whether to test vector retrieval
# =============================================================================

TEST_CASES = [

    # ── 1. Company Risk Queries ───────────────────────────────────────────────
    {
        "id"               : "R001",
        "query"            : "What are Amazon's main business risks?",
        "category"         : "company_risk",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["company_risk", "vector"],
        "relevant_keywords": ["risk", "fulfillment", "competition", "regulatory"],
        "min_results"      : 3,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "R002",
        "query"            : "What supply chain risks does Amazon disclose?",
        "category"         : "company_risk",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["company_risk", "vector"],
        "relevant_keywords": ["supply chain", "logistics", "fulfillment","shipping"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "R003",
        "query"            : "What regulatory risks does Amazon face?",
        "category"         : "company_risk",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["company_risk", "vector"],
        "relevant_keywords" :[
    "antitrust",
    "regulatory",
    "privacy",
    "compliance"
],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "R004",
        "query"            : "What cybersecurity risks does Amazon disclose?",
        "category"         : "company_risk",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["company_risk", "vector"],
        "relevant_keywords": ["cyber", "security", "breach", "data","aws","data"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 2. Cross-Company / Shared Risk Queries ────────────────────────────────
    {
        "id"               : "C001",
        "query"            : "Which companies are exposed to fuel cost risk?",
        "category"         : "shared_risk",
        "expected_tickers" : ["AMZN", "XOM", "GM"],
        "expected_types"   : ["shared_risk", "vector"],
        "relevant_keywords": ["fuel", "energy", "cost", "transportation"],
        "min_results"      : 5,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "C002",
        "query"            : "Which companies face geopolitical risk?",
        "category"         : "shared_risk",
        "expected_tickers" : ["AAPL", "NVDA", "MSFT"],
        "expected_types"   : ["shared_risk", "vector"],
        "relevant_keywords": ["geopolitical", "china", "trade", "sanction"],
        "min_results"      : 3,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "C003",
        "query"            : "Which pharma companies face regulatory risk?",
        "category"         : "shared_risk",
        "expected_tickers" : ["PFE", "JNJ", "ABBV", "MRK"],
        "expected_types"   : ["shared_risk", "vector"],
        "relevant_keywords": ["FDA", "regulatory", "approval", "compliance"],
        "min_results"      : 3,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "C004",
        "query"            : "Which energy companies are exposed to oil price risk?",
        "category"         : "shared_risk",
        "expected_tickers" : ["XOM", "CVX", "COP", "OXY"],
        "expected_types"   : ["shared_risk", "vector"],
        "relevant_keywords": ["oil", "crude", "commodity", "price"],
        "min_results"      : 3,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 3. Supply Chain Queries ───────────────────────────────────────────────
    {
        "id"               : "S001",
        "query"            : "What does Amazon depend on in its supply chain?",
        "category"         : "supply_chain",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["supply_chain_dependency", "vector"],
        "relevant_keywords": ["supplier", "fulfillment", "shipping", "warehouse"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "S002",
        "query"            : "What critical inputs does Amazon depend on?",
        "category"         : "supply_chain",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["supply_chain_dependency", "vector"],
        "relevant_keywords": ["fuel", "transportation", "shipping", "warehouse"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "S003",
        "query"            : "Amazon supply chain dependencies and critical inputs",
        "category"         : "supply_chain",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["supply_chain_dependency", "vector"],
        "relevant_keywords": ["fulfillment", "logistics", "distribution", "shipping","supplier"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 4. Propagation Queries ────────────────────────────────────────────────
    {
        "id"               : "P001",
        "query"            : "How would rising fuel prices affect Amazon?",
        "category"         : "propagation",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["propagation_risk", "vector"],
        "relevant_keywords": ["fuel", "transportation", "shipping", "logistics","cost"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "P002",
        "query"            : "What events could disrupt Amazon's fulfillment network?",
        "category"         : "propagation",
        "expected_tickers" : ["TSLA"],
        "expected_types"   : ["propagation_risk", "vector"],
        "relevant_keywords": ["port", "shipping", "supplier", "disruption"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "P003",
        "query"            : "How does port congestion propagate to Amazon operations?",
        "category"         : "propagation",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["propagation_risk", "vector"],
        "relevant_keywords": ["port", "logistics", "shipping", "delay"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 5. Competitor Queries ─────────────────────────────────────────────────
    {
        "id"               : "COMP001",
        "query"            : "Who are Amazon's main competitors?",
        "category"         : "competitor",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["competitor", "vector"],
        "relevant_keywords": ["cloud", "azure", "aws", "google", "compete"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "COMP002",
        "query"            : "Who competes with Amazon Web Services?",
        "category"         : "competitor",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["competitor", "vector"],
        "relevant_keywords": ["cloud", "azure", "aws", "google", "compete"],
        "min_results"      : 2,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 6. Macro Impact Queries ───────────────────────────────────────────────
    {
        "id"               : "M001",
        "query"            : "Which companies are most affected by rising oil prices?",
        "category"         : "macro_impact",
        "expected_tickers" : ["XOM", "CVX", "AMZN", "GM"],
        "expected_types"   : ["macro_impact", "shared_risk", "vector"],
        "relevant_keywords": ["oil", "fuel", "energy", "cost"],
        "min_results"      : 3,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "M002",
        "query"            : "How do rising interest rates affect tech companies?",
        "category"         : "macro_impact",
        "expected_tickers" : ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "expected_types"   : ["macro_impact", "vector"],
        "relevant_keywords": ["interest rate", "debt", "financing", "valuation"],
        "min_results"      : 3,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 7. News Queries ───────────────────────────────────────────────────────
    {
        "id"               : "N001",
        "query"            : "Recent news about Apple earnings",
        "category"         : "news",
        "expected_tickers" : ["AAPL"],
        "expected_types"   : ["news_mention", "vector"],
        "relevant_keywords": ["earnings", "revenue", "quarter", "results"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "N002",
        "query"            : "Breaking news affecting energy sector",
        "category"         : "news",
        "expected_tickers" : [],
        "expected_types"   : ["news_mention", "vector"],
        "relevant_keywords": ["energy", "oil", "gas", "production"],
        "min_results"      : 1,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 8. Executive Queries ──────────────────────────────────────────────────
    {
        "id"               : "E001",
        "query"            : "CEO changes in pharmaceutical companies",
        "category"         : "executive",
        "expected_tickers" : [],
        "expected_types"   : ["executive_change", "vector"],
        "relevant_keywords": ["CEO", "executive", "appointed", "departed"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "E002",
        "query"            : "Who are the key executives at Microsoft?",
        "category"         : "executive",
        "expected_tickers" : ["MSFT"],
        "expected_types"   : ["executive_change", "vector"],
        "relevant_keywords": ["CEO", "CFO", "Satya", "executive"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 9. Causal Chain Queries ───────────────────────────────────────────────
    {
        "id"               : "CA001",
        "query"            : "What caused margin compression at energy companies?",
        "category"         : "causal",
        "expected_tickers" : [],
        "expected_types"   : ["causal_chain", "vector"],
        "relevant_keywords": ["margin", "cost", "compression", "fuel", "inflation"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "CA002",
        "query"            : "Why did Amazon's logistics costs increase?",
        "category"         : "causal",
        "expected_tickers" : ["AMZN"],
        "expected_types"   : ["causal_chain", "vector"],
        "relevant_keywords": ["fuel", "labor", "logistics", "cost", "increase"],
        "min_results"      : 1,
        "k"                : 5,
        "use_graph"        : True,
        "use_vector"       : True,
    },

    # ── 10. Geographic Queries ────────────────────────────────────────────────
    {
        "id"               : "G001",
        "query"            : "Which companies have significant China exposure?",
        "category"         : "geographic",
        "expected_tickers" : ["AAPL", "NVDA"],
        "expected_types"   : ["geographic_exposure", "vector"],
        "relevant_keywords": ["china", "asia", "manufacturing", "revenue"],
        "min_results"      : 2,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },
    {
        "id"               : "G002",
        "query"            : "Companies concentrated in Middle East operations",
        "category"         : "geographic",
        "expected_tickers" : [],
        "expected_types"   : ["geographic_exposure", "vector"],
        "relevant_keywords": ["middle east", "oil", "operations", "region"],
        "min_results"      : 1,
        "k"                : 10,
        "use_graph"        : True,
        "use_vector"       : True,
    },
]


# =============================================================================
# Evaluation metrics
# =============================================================================

@dataclass
class QueryResult:
    """Result of evaluating one test case."""
    test_id       : str
    category      : str
    query         : str
    latency_ms    : float
    num_results   : int
    hit_rate      : bool
    recall_at_k   : float
    precision_at_k: float
    mrr           : float
    ndcg_at_k     : float
    errors        : list = field(default_factory=list)
    result_tickers: list = field(default_factory=list)
    result_types  : list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def recall_at_k(
    expected : list,
    retrieved: list,
    k        : int,
) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|
    Measures: did we find the expected entities in top K?
    """
    if not expected:
        return 1.0   # no expected = vacuously true
    top_k    = retrieved[:k]
    relevant = set(e.upper() for e in expected)
    found    = set(r.upper() for r in top_k if r)
    return len(relevant & found) / len(relevant)


def precision_at_k(
    expected : list,
    retrieved: list,
    k        : int,
) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / K
    Measures: of top K, how many are relevant?
    """
    if not expected or not retrieved:
        return 0.0
    top_k    = retrieved[:k]
    relevant = set(e.upper() for e in expected)
    found    = sum(1 for r in top_k if r and r.upper() in relevant)
    return found / min(k, len(top_k))


def mean_reciprocal_rank(
    expected : list,
    retrieved: list,
) -> float:
    """
    MRR = 1 / rank_of_first_relevant_result
    Measures: how early does the first relevant result appear?
    """
    if not expected:
        return 1.0
    relevant = set(e.upper() for e in expected)
    for rank, r in enumerate(retrieved, start=1):
        if r and r.upper() in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    expected : list,
    retrieved: list,
    k        : int,
) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain.
    Rewards finding relevant results early in the ranking.
    """
    if not expected:
        return 1.0
    relevant = set(e.upper() for e in expected)
    top_k    = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, r in enumerate(top_k):
        if r and r.upper() in relevant:
            dcg += 1.0 / math.log2(i + 2)  # log2(rank+1)

    # Ideal DCG — all relevant items at top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# Test runner
# =============================================================================

class RetrievalEvaluator:

    def __init__(self):
        self.graph   = GraphRetriever()
        self.vector  = VectorRetriever()
        self.merger  = RetrievalMerger()
        self.parser  = QueryParser()

    def run_query(self, test: dict) -> QueryResult:
        """Run one test case and compute metrics."""
        t0 = time.time()
        errors = []

        try:
            pq = self.parser.parse(test["query"], top_k=test.get("k", 5) * 2)

            graph_results  = []
            vector_results = []

            # Graph retrieval
            if test.get("use_graph", True) and pq.use_graph:
                try:
                    graph_results = self._run_graph(pq)
                except Exception as e:
                    errors.append(f"Graph error: {str(e)[:80]}")

            # Vector retrieval
            if test.get("use_vector", True):
                try:
                    vector_results = self.vector.retrieve(
                        query        = test["query"],
                        tickers      = pq.tickers if pq.tickers else None,
                        sections     = pq.sections if pq.sections else None,
                        filing_types = pq.filing_types if pq.filing_types else None,
                        date_from    = pq.date_from,
                        date_to      = pq.date_to,
                        top_k        = test.get("k", 5) * 3,
                    )
                except Exception as e:
                    errors.append(f"Vector error: {str(e)[:80]}")

            # Merge
            merged = self.merger.merge(
                graph_results,
                vector_results,
                top_k=test.get("k", 5),
            )

        except Exception as e:
            errors.append(f"Fatal: {str(e)[:80]}")
            merged = []

        latency_ms = (time.time() - t0) * 1000

        # Extract tickers and types from results
        result_tickers = list(dict.fromkeys(
            r.primary_entity for r in merged if r.primary_entity
        ))
        result_types = list(dict.fromkeys(
            r.result_type for r in merged
        ))

        k          = test.get("k", 5)
        expected   = test.get("expected_tickers", [])
        top_k_tick = result_tickers[:k]

        return QueryResult(
            test_id        = test["id"],
            category       = test["category"],
            query          = test["query"],
            latency_ms     = round(latency_ms, 1),
            num_results    = len(merged),
            hit_rate       = len(merged) >= test.get("min_results", 1),
            recall_at_k    = round(recall_at_k(expected, top_k_tick, k), 3),
            precision_at_k = round(precision_at_k(expected, top_k_tick, k), 3),
            mrr            = round(mean_reciprocal_rank(expected, top_k_tick), 3),
            ndcg_at_k      = round(ndcg_at_k(expected, top_k_tick, k), 3),
            errors         = errors,
            result_tickers = result_tickers[:10],
            result_types   = result_types,
        )

    def _run_graph(self, pq) -> list:
        """Route to correct graph retrieval method based on parsed intent."""
        method = pq.graph_query_type
        ticker = pq.tickers[0] if pq.tickers else None

        if method == "get_company_risks" and ticker:
            return self.graph.get_company_risks(ticker)
        if method == "get_shared_risk_companies":
            cat = pq.risk_categories[0] if pq.risk_categories else "macro_risk"
            return self.graph.get_shared_risk_companies(cat)
        if method == "get_supply_chain" and ticker:
            return self.graph.get_supply_chain(ticker)
        if method == "get_propagation_risks" and ticker:
            return self.graph.get_propagation_risks(ticker)
        if method == "get_competitors" and ticker:
            return self.graph.get_competitors(ticker)
        if method == "get_causal_chains":
            kw = pq.keywords[0] if pq.keywords else ""
            return self.graph.get_causal_chains(kw)
        if method == "get_macro_impact":
            ind = pq.macro_indicators[0] if pq.macro_indicators else "oil"
            return self.graph.get_macro_impact(ind)
        if method == "get_news_impact" and ticker:
            return self.graph.get_news_impact(ticker)
        if method == "get_executive_changes":
            return self.graph.get_executive_changes(ticker)
        if method == "get_geographic_exposure":
            geo = pq.geographies[0] if pq.geographies else ""
            return self.graph.get_geographic_exposure(geo)
        if method == "get_market_signals" and ticker:
            return self.graph.get_market_signals(ticker)
        if method == "get_company_overview" and ticker:
            overview = self.graph.get_company_overview(ticker)
            return []  # overview is a dict, not list
        return []

    def evaluate(
        self,
        test_cases : list,
        output_path: Optional[Path] = None,
    ) -> dict:
        """Run all test cases and compute aggregate metrics."""

        print("\n" + "=" * 80)
        print("RETRIEVAL EVALUATION")
        print("=" * 80)
        print(f"Running {len(test_cases)} test cases...\n")

        results     = []
        by_category = defaultdict(list)

        for i, test in enumerate(test_cases):
            print(f"[{i+1:02d}/{len(test_cases)}] {test['id']} — {test['query'][:60]}")
            r = self.run_query(test)
            results.append(r)
            by_category[r.category].append(r)

            status = "✅" if r.hit_rate and not r.errors else "❌"
            print(f"         {status} "
                  f"results={r.num_results} "
                  f"R@{test.get('k',5)}={r.recall_at_k:.2f} "
                  f"MRR={r.mrr:.2f} "
                  f"NDCG={r.ndcg_at_k:.2f} "
                  f"latency={r.latency_ms:.0f}ms")
            if r.errors:
                for e in r.errors:
                    print(f"         ⚠  {e}")
            if r.result_tickers:
                print(f"         → tickers: {r.result_tickers[:5]}")

        # Aggregate metrics
        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        overall = {
            "total_tests"     : len(results),
            "hit_rate"        : avg([1.0 if r.hit_rate else 0.0 for r in results]),
            "recall_at_k"     : avg([r.recall_at_k    for r in results]),
            "precision_at_k"  : avg([r.precision_at_k for r in results]),
            "mrr"             : avg([r.mrr             for r in results]),
            "ndcg_at_k"       : avg([r.ndcg_at_k       for r in results]),
            "avg_latency_ms"  : avg([r.latency_ms       for r in results]),
            "p50_latency_ms"  : round(sorted([r.latency_ms for r in results])
                                      [len(results)//2], 1),
            "p95_latency_ms"  : round(sorted([r.latency_ms for r in results])
                                      [int(len(results)*0.95)], 1),
            "error_rate"      : avg([1.0 if r.errors else 0.0 for r in results]),
            "tests_with_errors": sum(1 for r in results if r.errors),
        }

        by_category_summary = {}
        for cat, cat_results in by_category.items():
            by_category_summary[cat] = {
                "count"         : len(cat_results),
                "hit_rate"      : avg([1.0 if r.hit_rate else 0.0 for r in cat_results]),
                "recall_at_k"   : avg([r.recall_at_k    for r in cat_results]),
                "mrr"           : avg([r.mrr             for r in cat_results]),
                "ndcg_at_k"     : avg([r.ndcg_at_k       for r in cat_results]),
                "avg_latency_ms": avg([r.latency_ms       for r in cat_results]),
            }

        # Print summary
        print("\n" + "=" * 80)
        print("AGGREGATE METRICS")
        print("=" * 80)
        print(f"  Total tests      : {overall['total_tests']}")
        print(f"  Hit Rate         : {overall['hit_rate']:.1%}")
        print(f"  Recall@K         : {overall['recall_at_k']:.3f}")
        print(f"  Precision@K      : {overall['precision_at_k']:.3f}")
        print(f"  MRR              : {overall['mrr']:.3f}")
        print(f"  NDCG@K           : {overall['ndcg_at_k']:.3f}")
        print(f"  Avg Latency      : {overall['avg_latency_ms']:.0f}ms")
        print(f"  P50 Latency      : {overall['p50_latency_ms']:.0f}ms")
        print(f"  P95 Latency      : {overall['p95_latency_ms']:.0f}ms")
        print(f"  Error Rate       : {overall['error_rate']:.1%}")

        print("\n  By Category:")
        for cat, metrics in sorted(by_category_summary.items()):
            print(f"    {cat:<20} "
                  f"n={metrics['count']} "
                  f"R@K={metrics['recall_at_k']:.2f} "
                  f"MRR={metrics['mrr']:.2f} "
                  f"NDCG={metrics['ndcg_at_k']:.2f} "
                  f"lat={metrics['avg_latency_ms']:.0f}ms")

        # Identify weak spots
        weak = [r for r in results
                if r.recall_at_k < 0.5 or r.mrr < 0.3 or not r.hit_rate]
        if weak:
            print(f"\n  ⚠  Weak queries ({len(weak)}) — need improvement:")
            for r in weak:
                print(f"    {r.test_id} [{r.category}] "
                      f"R@K={r.recall_at_k} MRR={r.mrr} "
                      f"— {r.query[:60]}")

        # Save results
        report = {
            "overall"            : overall,
            "by_category"        : by_category_summary,
            "individual_results" : [r.to_dict() for r in results],
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2))
            print(f"\n  Report saved to: {output_path}")

        return report


# =============================================================================
# Main
# =============================================================================

def main():
    args        = sys.argv[1:]
    quick       = "--quick"       in args
    graph_only  = "--graph-only"  in args
    vector_only = "--vector-only" in args
    category    = None

    for i, a in enumerate(args):
        if a == "--category" and i + 1 < len(args):
            category = args[i + 1]

    # Filter test cases
    tests = TEST_CASES
    if quick:
        tests = TEST_CASES[:5]
    if category:
        tests = [t for t in tests if t["category"] == category]
    if graph_only:
        for t in tests:
            t["use_vector"] = False
    if vector_only:
        for t in tests:
            t["use_graph"] = False

    evaluator = RetrievalEvaluator()
    report    = evaluator.evaluate(
        tests,
        output_path=Path("data/eval/retrieval_report.json"),
    )

    # Exit code: 0 if overall Recall@K >= 0.5, else 1
    overall_recall = report["overall"]["recall_at_k"]
    sys.exit(0 if overall_recall >= 0.5 else 1)


if __name__ == "__main__":
    main()