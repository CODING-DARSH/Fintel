# =============================================================================
# src/agents/gather_evidence.py
# =============================================================================
# STEP 3 of the agent build: evidence gathering for ONE sub-question.
#
# Reuses the existing retrieval stack as-is — no new retrieval logic:
#   QueryParser.parse()      -> intent + entities from the sub-question text
#   GraphRetriever           -> same dispatch pattern as test_retrieval.py's
#                               _run_graph() helper
#   VectorRetriever.retrieve -> vector/filing chunk search
#   RetrievalMerger.merge()  -> the same merge() already validated in eval
#
# Output: list[Evidence], one per MergedResult, with reliability/recency/
# relevance/confidence filled in. Evidence.text is the FULL merged text/
# summary, never truncated (future UI needs the real chunk).
#
# NOTE: relevance scoring here is a placeholder (based on retrieval rank,
# not semantic re-scoring) — flagged clearly below. A real relevance model
# is a later step, not blocking this one.
# =============================================================================

from __future__ import annotations

import logging
from datetime import date, datetime

from src.retrieval import GraphRetriever, VectorRetriever, RetrievalMerger, QueryParser
from src.agents.models.contract import SubQuestion, Evidence, SourceRef, PropagationResult
from src.agents.analytics.graph_propagation import find_paths, to_evidence

log = logging.getLogger(__name__)

try:
    from config import SOURCE_TRUST
except Exception:
    # Fallback if config isn't importable in this context — keeps this
    # module usable standalone / in tests without the full project config.
    SOURCE_TRUST = {"sec_filing": 1.00, "general_news": 0.60,
                     "market_data": 0.85, "macro_data": 0.75}


# ---------------------------------------------------------------------------
# Graph dispatch — uses GraphRetriever.query() generic dispatcher.
#
# We no longer hardcode which pq.graph_query_type values map to which
# method — GraphRetriever.query() looks up the method by name and filters
# kwargs to whatever that method actually declares. This means a new
# get_* method added to GraphRetriever becomes callable from here with
# zero changes to this file, as long as ParsedQuery's graph_query_type
# produces the matching method name (e.g. "get_regulatory_impact").
#
# We still build a broad set of CANDIDATE kwargs from the parsed query,
# since different graph methods use different param names for similar
# concepts (ticker vs cause_keyword vs indicator vs geography). Passing
# all candidates is safe — query() drops whatever a given method doesn't
# declare.
# ---------------------------------------------------------------------------

def _run_graph(graph: GraphRetriever, pq) -> list:
    candidate_kwargs = {
        "ticker"          : pq.tickers[0] if pq.tickers else None,
        "risk_category"   : pq.risk_categories[0] if pq.risk_categories else None,
        "cause_keyword"   : pq.keywords[0] if pq.keywords else None,
        "indicator"       : pq.macro_indicators[0] if pq.macro_indicators else None,
        "geography"       : pq.geographies[0] if pq.geographies else None,
    }
    return graph.query(pq.graph_query_type, **candidate_kwargs)


# ---------------------------------------------------------------------------
# Evidence scoring helpers
# ---------------------------------------------------------------------------

# Maps GraphRetriever's granular result_type (set per-method in
# graph_retriever.py, e.g. "news_mention", "macro_impact") to the
# SOURCE_TRUST key that actually reflects where that data came from.
# BUG THIS FIXES: _reliability_for previously always returned
# SOURCE_TRUST["sec_filing"] for every graph result, regardless of
# whether it actually originated from a filing-derived node (Risk,
# Input via DEPENDS_ON) or a connector-derived node (NewsArticle,
# MacroSignal, MarketSignal) — meaning news-sourced evidence was being
# scored with SEC-filing-grade reliability every single time, silently
# defeating the whole point of having differentiated SOURCE_TRUST
# values. Only became visible as a real problem once news/macro data
# actually started flowing into the graph in volume.
GRAPH_RESULT_TYPE_TRUST = {
    "company_risk"           : "sec_filing",
    "shared_risk"            : "sec_filing",
    "supply_chain_dependency": "sec_filing",
    "propagation_risk"       : "sec_filing",
    "competitor"             : "sec_filing",
    "causal_chain"           : "sec_filing",
    "executive_change"       : "sec_filing",
    "geographic_exposure"    : "sec_filing",
    "macro_impact"           : "macro_data",
    "market_signal"          : "market_data",
    "news_mention"           : "general_news",
}


def _reliability_for(merged_result) -> float:
    """
    Reliability by ACTUAL source type. Vector results are always filing
    chunks (VectorRetriever only ever indexes data/extracted/ chunk
    text — confirmed by embedder.py/chunker.py), so sec_filing is
    correct there. Graph results now look up the granular result_type
    (news_mention, macro_impact, etc.) via GRAPH_RESULT_TYPE_TRUST
    instead of assuming sec_filing for everything — that assumption was
    silently wrong for any graph evidence sourced from a connector
    rather than a filing.
    """
    if merged_result.result_type == "graph":
        src = merged_result.sources[0] if merged_result.sources else {}
        granular_type = src.get("result_type", "")
        trust_key = GRAPH_RESULT_TYPE_TRUST.get(granular_type, "sec_filing")
        return SOURCE_TRUST.get(trust_key, 0.9)
    return SOURCE_TRUST.get("sec_filing", 1.0)


def _recency_for(merged_result) -> str:
    if merged_result.filing_dates:
        return merged_result.filing_dates[0]
    return "current"


def _relevance_for(rank: int, total: int) -> float:
    """
    PLACEHOLDER: rank-based relevance, not semantic relevance.
    RetrievalMerger already rank-orders by RRF score, so rank position is
    a reasonable proxy for now. Replace with a real relevance judgment
    (e.g. cross-encoder or LLM-scored) once evidence evaluation is built
    out as its own step — not done here to avoid scope creep in this file.
    """
    if total <= 1:
        return 1.0
    return round(1.0 - (rank / total) * 0.5, 3)  # ranges 1.0 -> 0.5


def _to_source_ref(merged_result) -> SourceRef:
    src = merged_result.sources[0] if merged_result.sources else {}
    if merged_result.result_type == "graph":
        return SourceRef(
            type="graph",
            result_type=src.get("result_type"),
            chunks=src.get("chunks", []),
        )
    return SourceRef(
        type="vector",
        chunk_id=src.get("chunk_id"),
        filing_id=src.get("filing_id"),
    )


# ---------------------------------------------------------------------------
# Propagation branch — EXISTENCE-VERIFIED start-entity resolution.
#
# Earlier version gated propagation on whether the model happened to
# label a sub-question's retrieval_focus as "propagation". A real run
# showed this is NOT reliable: given an explicitly propagation-shaped
# question (a China graphite export ban affecting automakers), the model
# labeled its own sub-questions "regulatory", "macro", "supply_chain",
# "filing" — never "propagation" — because those labels were also
# accurate descriptions of what each sub-question needed. Trusting the
# label meant propagation never ran at all on the exact case it was
# built for.
#
# Fix: stop gating on the label. Instead, for EVERY sub-question,
# opportunistically try to resolve a start entity from whatever
# geography/keywords ParsedQuery extracted, then VERIFY each candidate
# actually exists as a node in the graph (GraphRetriever.node_exists)
# before running any traversal. This is airtight in a way label-matching
# never was:
#   - No dependence on the model choosing the "right" word for a focus
#     label — propagation is attempted based on what's actually in the
#     query, for every sub-question, every time.
#   - No risk of the earlier "everything"/"fine" nonsense-keyword
#     failure — a candidate that isn't a real graph node is rejected
#     before any traversal or evidence is produced, not after.
#   - Tries geography candidates AND every keyword candidate (not just
#     the first), stopping at the first one that verifiably exists.
# ---------------------------------------------------------------------------

def _resolve_propagation_start(pq, graph: GraphRetriever) -> Optional[tuple[str, str, str]]:
    """
    Returns (start_label, start_id, start_entity_name) for the first
    candidate — checking geographies before keywords, since a named
    country is a stronger signal of an intended disruption anchor than
    an arbitrary extracted keyword — that VERIFIABLY EXISTS in the graph.
    Returns None if nothing extracted from the query corresponds to a
    real node, which is the normal, expected outcome for most
    non-propagation-shaped questions.
    """
    candidates: list[tuple[str, str, str]] = []
    for geo in pq.geographies:
        candidates.append(("Geography", geo.lower().replace(" ", "_"), geo))
    for kw in pq.keywords:
        candidates.append(("Input", kw.lower().replace(" ", "_"), kw))

    for start_label, start_id, start_name in candidates:
        try:
            if graph.node_exists(start_label, start_id):
                return (start_label, start_id, start_name)
        except Exception as e:
            log.warning(f"node_exists check raised for {start_label}:{start_id}: {e}")
            continue

    return None


def _gather_propagation_evidence(
    sub_question: SubQuestion, pq, graph: GraphRetriever, k: int,
    propagation_cache: Optional[dict] = None,
) -> list[Evidence]:
    start = _resolve_propagation_start(pq, graph)
    if start is None:
        # Normal, expected outcome for most questions — no propagation-
        # shaped entity was found in this query, or nothing extracted
        # actually corresponds to a real graph node. Not an error.
        return []

    start_label, start_id, start_name = start
    target_ticker = pq.tickers[0] if pq.tickers else None

    # cache key deliberately does NOT include target_ticker — the
    # traversal from a given start entity finds ALL reachable companies
    # regardless of target, so the same PropagationResult is valid and
    # reusable across sub-questions even if they name different (or no)
    # target ticker. target_found is just a lookup into an already-
    # computed companies_reached dict, so it's still correct per-caller.
    cache_key = (start_label, start_id)

    if propagation_cache is not None and cache_key in propagation_cache:
        result = propagation_cache[cache_key]
        log.info(
            f"[{sub_question.id}] propagation reused from cache: "
            f"{start_label}:{start_id} (no re-traversal)"
        )
    else:
        try:
            result = find_paths(
                graph, start_label, start_id, start_entity_name=start_name,
                target_ticker=target_ticker, max_depth=4,
            )
        except Exception as e:
            log.warning(f"[{sub_question.id}] propagation search failed: {e}")
            return []

        log.info(
            f"[{sub_question.id}] propagation start entity verified: "
            f"{start_label}:{start_id} — {len(result.companies_reached)} "
            f"companies reached"
        )

        if propagation_cache is not None:
            propagation_cache[cache_key] = result

    # target_found/target_ticker are per-caller even when the underlying
    # traversal is cached — re-derive them from the cached result rather
    # than trusting whatever target_ticker was baked in when it was
    # first computed by a possibly-different sub-question.
    if target_ticker:
        result = PropagationResult(
            start_entity=result.start_entity,
            start_entity_type=result.start_entity_type,
            companies_reached=result.companies_reached,
            paths=result.paths,
            target_ticker=target_ticker.upper(),
            target_found=target_ticker.upper() in result.paths,
        )

    ev = to_evidence(result, sub_question.id)
    return [ev] if ev else []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gather_evidence(
    sub_question: SubQuestion,
    k: int = 5,
    graph: GraphRetriever | None = None,
    vector: VectorRetriever | None = None,
    merger: RetrievalMerger | None = None,
    parser: QueryParser | None = None,
    propagation_cache: Optional[dict] = None,
) -> list[Evidence]:
    """
    Run retrieval for ONE sub-question and return scored Evidence objects.

    retriever/merger/parser instances are optional params (not re-created
    per call) so a caller running many sub-questions in one orchestrator
    pass can share instances instead of reconnecting to Neo4j/Chroma each time.

    propagation_cache: optional dict shared across multiple gather_evidence
    calls within the same hop, keyed by (start_label, start_id). Prevents
    re-running the same expensive graph traversal when multiple
    sub-questions resolve to the same start entity — a real run showed
    three sub-questions all independently re-running the identical
    "china" traversal and producing byte-identical evidence. Pass None
    (default) to disable caching.
    """
    graph  = graph  or GraphRetriever()
    vector = vector or VectorRetriever()
    merger = merger or RetrievalMerger()
    parser = parser or QueryParser()

    pq = parser.parse(sub_question.text, top_k=k * 2)

    graph_results = []
    if pq.use_graph:
        try:
            graph_results = _run_graph(graph, pq)
        except Exception as e:
            log.warning(f"[{sub_question.id}] graph retrieval failed: {e}")

    vector_results = []
    try:
        vector_results = vector.retrieve(
            query        = sub_question.text,
            tickers      = pq.tickers if pq.tickers else None,
            sections     = pq.sections if pq.sections else None,
            filing_types = pq.filing_types if pq.filing_types else None,
            date_from    = pq.date_from,
            date_to      = pq.date_to,
            top_k        = k * 3,
        )
    except Exception as e:
        log.warning(f"[{sub_question.id}] vector retrieval failed: {e}")

    merged = merger.merge(graph_results, vector_results, top_k=k)

    total = len(merged)
    evidence: list[Evidence] = []
    for i, mr in enumerate(merged):
        evidence.append(
            Evidence(
                id=f"{sub_question.id}_ev{i+1}",
                sub_question_id=sub_question.id,
                source=_to_source_ref(mr),
                text=mr.text,  # full text, not truncated
                primary_entity=mr.primary_entity,
                filing_dates=mr.filing_dates,
                reliability=_reliability_for(mr),
                recency=_recency_for(mr),
                relevance=_relevance_for(i, total),
                confidence=round(mr.final_score, 3),
            )
        )

    if not evidence:
        log.info(f"[{sub_question.id}] no evidence found for: {sub_question.text!r}")

    # Propagation is attempted OPPORTUNISTICALLY for every sub-question,
    # not gated by retrieval_focus label (see _resolve_propagation_start's
    # docstring for why label-gating was removed — it wasn't reliable).
    # Cost is low when nothing plausible exists: _resolve_propagation_start
    # returns None immediately unless a real graph node is found, and this
    # stays additive to the normal merge results either way.
    prop_evidence = _gather_propagation_evidence(sub_question, pq, graph, k, propagation_cache)
    evidence.extend(prop_evidence)

    return evidence


if __name__ == "__main__":
    # Manual structural check — this WILL fail to connect to Neo4j/Chroma
    # if they're not running, which is expected. It's here to confirm the
    # function signature and control flow, not to be a live integration test.
    sq = SubQuestion(id="sq1", text="What supply chain risks does Amazon disclose?",
                      retrieval_focus="filing")
    try:
        result = gather_evidence(sq, k=3)
        print(f"Got {len(result)} evidence items")
        for e in result:
            print(e.model_dump())
    except Exception as e:
        print(f"Expected if DBs aren't running locally: {e}")