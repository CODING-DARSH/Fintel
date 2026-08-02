# =============================================================================
# src/agents/analytics/graph_propagation.py
# =============================================================================
# Analytics layer — sits AFTER retrieval, not part of it (per the
# retrieval-vs-analytics separation discussed). Consumes raw path data
# from GraphRetriever.get_propagation_paths() (structure only, no
# scoring) and produces scored PropagationPath / PropagationResult
# objects, then converts those into Evidence objects that the rest of
# the pipeline (gap detection, contradiction detection, hypothesis,
# synthesis) already knows how to consume — they don't know or care
# this evidence came from graph traversal rather than a retrieval merge.
#
# Three functions, matching the three responsibilities discussed:
#   score_path()   — pure arithmetic over one path's edges
#   find_paths()   — calls GraphRetriever, builds a PropagationResult
#   to_evidence()  — converts a PropagationResult into Evidence,
#                    INCLUDING the explicit "not found" case
#
# No LLM call anywhere in this file — this is deterministic graph
# arithmetic, same category of work as gather_all_evidence.py's
# detect_gaps().
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional

from src.retrieval.graph_retriever import GraphRetriever
from src.agents.models.contract import (
    PathStep, PropagationPath, PropagationResult, Evidence, SourceRef,
)

log = logging.getLogger(__name__)

try:
    from config import PROPAGATION_CATEGORICAL_WEIGHTS, PROPAGATION_RELATION_DEFAULTS
except Exception:
    # Fallback so this module stays importable/testable standalone even
    # if config.py isn't reachable in a given context — mirrors the same
    # defensive pattern used in gather_evidence.py for SOURCE_TRUST.
    PROPAGATION_CATEGORICAL_WEIGHTS = {"high": 0.85, "medium": 0.5, "low": 0.2}
    PROPAGATION_RELATION_DEFAULTS = {
        "DEPENDS_ON": 0.5, "SUPPLIES_TO": 0.5, "SOURCED_FROM": 0.6,
        "EXPOSED_TO": 0.5, "PROPAGATES_TO": 0.6, "BUYS_FROM": 0.4,
    }

CATEGORICAL_FIELDS = ("criticality", "severity", "dependency_level")
NUMERIC_FIELDS = ("percentage", "cost_share")

# Below this, a path is considered too weak to surface as "found" —
# still returned in the result (never silently dropped), but this
# threshold is what to_evidence() uses to decide the wording/framing
# of the resulting Evidence text. Uncalibrated, same caveat as the
# weight tables above.
WEAK_PATH_SCORE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# score_path — pure arithmetic, no DB, no LLM
# ---------------------------------------------------------------------------

def _resolve_edge_weight(rel_data: dict) -> tuple[float, str]:
    """
    Resolution priority (documented in config.py alongside the tables):
      1. real numeric field (percentage as 0-100 -> /100, or cost_share
         if already 0-1)
      2. categorical field mapped through PROPAGATION_CATEGORICAL_WEIGHTS
      3. flat per-relation-type default

    Returns (weight, weight_source) — weight_source is stored on the
    PathStep so it's always visible WHY a given number was used, not
    just what the number is.
    """
    pct = rel_data.get("percentage")
    if isinstance(pct, (int, float)) and pct is not None:
        return (max(0.0, min(1.0, pct / 100.0)), "numeric_field")

    cost_share = rel_data.get("cost_share")
    if isinstance(cost_share, (int, float)) and cost_share is not None:
        # cost_share may be stored as a fraction already, or as a
        # percentage — normalize defensively rather than assume.
        val = cost_share / 100.0 if cost_share > 1.0 else cost_share
        return (max(0.0, min(1.0, val)), "numeric_field")

    for field in CATEGORICAL_FIELDS:
        val = rel_data.get(field)
        if isinstance(val, str) and val.lower() in PROPAGATION_CATEGORICAL_WEIGHTS:
            return (PROPAGATION_CATEGORICAL_WEIGHTS[val.lower()], "categorical_map")

    rel_type = rel_data.get("type", "")
    default = PROPAGATION_RELATION_DEFAULTS.get(rel_type, 0.4)
    return (default, "relation_default")


def _build_path_steps(path_nodes: list[dict], path_rels: list[dict]) -> list[PathStep]:
    """
    path_nodes / path_rels come straight from get_propagation_paths()'s
    Cypher output. Zips them into typed PathStep objects, resolving each
    edge's weight via _resolve_edge_weight().
    """
    steps = []
    for i, rel in enumerate(path_rels):
        from_node = path_nodes[i] if i < len(path_nodes) else {}
        to_node = path_nodes[i + 1] if i + 1 < len(path_nodes) else {}
        weight, source = _resolve_edge_weight(rel)
        steps.append(PathStep(
            from_node=from_node.get("id") or from_node.get("name") or "?",
            from_type=(from_node.get("labels") or ["?"])[0],
            relation=rel.get("type", "?"),
            to_node=to_node.get("id") or to_node.get("name") or "?",
            to_type=(to_node.get("labels") or ["?"])[0],
            edge_confidence=rel.get("confidence"),
            edge_weight=weight,
            weight_source=source,
        ))
    return steps


def score_path(steps: list[PathStep]) -> float:
    """
    Multiply edge_confidence * edge_weight along every step. Missing
    edge_confidence uses 0.7 (not 1.0, not 0.0 — matches the extractor's
    own default confidence for risk_signals/dependencies in
    graph_loader.py, so an unscored edge doesn't silently look either
    perfectly certain or worthless).
    """
    score = 1.0
    for step in steps:
        conf = step.edge_confidence if step.edge_confidence is not None else 0.7
        score *= conf * step.edge_weight
    return round(score, 4)


# ---------------------------------------------------------------------------
# find_paths — calls GraphRetriever, builds PropagationResult
# ---------------------------------------------------------------------------

def find_paths(
    graph: GraphRetriever,
    start_label: str,
    start_id: str,
    start_entity_name: Optional[str] = None,
    target_ticker: Optional[str] = None,
    max_depth: int = 4,
) -> PropagationResult:
    """
    Search outward from (start_label, start_id) for all reachable
    companies, score every path found, and return a PropagationResult.

    target_ticker is optional — if given, target_found is set explicitly
    based on whether that ticker was actually reached. If not given,
    this just returns everything reachable (useful for "who's affected
    by X" questions with no single named target).
    """
    raw_rows = graph.get_propagation_paths(start_label, start_id, max_depth=max_depth)

    paths_by_ticker: dict[str, list[PropagationPath]] = {}
    for row in raw_rows:
        ticker = row.get("ticker")
        if not ticker:
            continue
        steps = _build_path_steps(row.get("path_nodes", []), row.get("path_rels", []))
        if not steps:
            continue
        score = score_path(steps)
        paths_by_ticker.setdefault(ticker, []).append(PropagationPath(
            steps=steps,
            target_ticker=ticker,
            path_score=score,
            depth=row.get("depth", len(steps)),
        ))

    # sort each ticker's paths strongest-first, so to_evidence() can just
    # take [0] for the best path without re-sorting
    for ticker in paths_by_ticker:
        paths_by_ticker[ticker].sort(key=lambda p: p.path_score, reverse=True)

    companies_reached = list(paths_by_ticker.keys())
    target_found = target_ticker.upper() in paths_by_ticker if target_ticker else False

    return PropagationResult(
        start_entity=start_entity_name or start_id,
        start_entity_type=start_label,
        companies_reached=companies_reached,
        paths=paths_by_ticker,
        target_ticker=target_ticker.upper() if target_ticker else None,
        target_found=target_found,
    )


# ---------------------------------------------------------------------------
# to_evidence — including the explicit "not found" case
# ---------------------------------------------------------------------------

def _describe_path(path: PropagationPath) -> str:
    parts = []
    for step in path.steps:
        parts.append(f"{step.from_node} --[{step.relation}]--> {step.to_node}")
    return "; ".join(parts)


def to_evidence(result: PropagationResult, sub_question_id: str) -> Optional[Evidence]:
    """
    Convert a PropagationResult into ONE Evidence object.

    If target_ticker was given but not found: still returns an Evidence
    object (not None) stating plainly that no documented chain was
    found — absence is evidence and must reach the hypothesis step, not
    silently vanish.

    If no target_ticker was given at all: summarizes all companies
    reached instead (no single "found/not found" framing applies).
    """
    source = SourceRef(type="graph", result_type="propagation", chunks=[])

    if result.target_ticker and not result.target_found:
        return Evidence(
            id=f"{sub_question_id}_propagation",
            sub_question_id=sub_question_id,
            source=source,
            text=(
                f"No documented dependency chain was found connecting "
                f"{result.start_entity} to {result.target_ticker} within the "
                f"searched depth. This reflects what has been extracted from "
                f"filings so far, not a claim that no real-world connection exists."
            ),
            primary_entity=result.target_ticker,
            reliability=0.5,   # neither confirms nor denies — flagged as such in text
            recency="current",
            relevance=0.5,
            confidence=0.0,
        )

    if result.target_ticker and result.target_found:
        best_path = result.paths[result.target_ticker][0]
        n_paths = len(result.paths[result.target_ticker])
        extra = f" ({n_paths - 1} additional weaker path(s) also found)" if n_paths > 1 else ""
        return Evidence(
            id=f"{sub_question_id}_propagation",
            sub_question_id=sub_question_id,
            source=source,
            text=(
                f"A documented connection pattern across extracted disclosures links "
                f"{result.start_entity} to {result.target_ticker}: {_describe_path(best_path)}. "
                f"Path confidence {best_path.path_score:.2f} — this reflects connection "
                f"strength given available data, not a predictive claim.{extra}"
            ),
            primary_entity=result.target_ticker,
            reliability=0.8,
            recency="current",
            relevance=0.9 if best_path.path_score >= WEAK_PATH_SCORE_THRESHOLD else 0.4,
            confidence=best_path.path_score,
        )

    if not result.companies_reached:
        return Evidence(
            id=f"{sub_question_id}_propagation",
            sub_question_id=sub_question_id,
            source=source,
            text=f"No companies were found connected to {result.start_entity} in the graph.",
            primary_entity=result.start_entity,
            reliability=0.5,
            recency="current",
            relevance=0.5,
            confidence=0.0,
        )

    top = sorted(
        result.companies_reached,
        key=lambda t: result.paths[t][0].path_score,
        reverse=True,
    )[:5]
    summary = "; ".join(f"{t} ({result.paths[t][0].path_score:.2f})" for t in top)
    return Evidence(
        id=f"{sub_question_id}_propagation",
        sub_question_id=sub_question_id,
        source=source,
        text=(
            f"Companies with a documented connection to {result.start_entity} "
            f"(top {len(top)} of {len(result.companies_reached)} found, by path "
            f"confidence): {summary}."
        ),
        primary_entity=result.start_entity,
        reliability=0.8,
        recency="current",
        relevance=0.8,
        confidence=result.paths[top[0]][0].path_score if top else 0.0,
    )


if __name__ == "__main__":
    # Structural smoke test with fake raw rows — no live Neo4j needed,
    # mirrors the shape get_propagation_paths() actually returns.
    fake_rows_found = [
        {
            "ticker": "TSLA",
            "depth": 3,
            "path_nodes": [
                {"labels": ["Input"], "id": "graphite", "name": "Graphite"},
                {"labels": ["Company"], "id": "PANASONIC", "name": "Panasonic"},
                {"labels": ["Company"], "id": "TSLA", "name": "Tesla"},
            ],
            "path_rels": [
                {"type": "DEPENDS_ON", "confidence": 0.8, "criticality": "high",
                 "severity": None, "dependency_level": None, "cost_share": None, "percentage": None},
                {"type": "SUPPLIES_TO", "confidence": 0.75, "criticality": None,
                 "severity": None, "dependency_level": None, "cost_share": None, "percentage": None},
            ],
        }
    ]

    class StubGraph(GraphRetriever):
        def __init__(self):
            pass
        def get_propagation_paths(self, start_label, start_id, max_depth=4, path_limit=50):
            return fake_rows_found if start_id == "graphite" else []

    print("--- target found ---")
    result = find_paths(StubGraph(), "Input", "graphite", "graphite", target_ticker="TSLA")
    print(result.model_dump())
    ev = to_evidence(result, "sq_propagation")
    print(ev.model_dump())

    print("\n--- target NOT found ---")
    result2 = find_paths(StubGraph(), "Input", "lithium", "lithium", target_ticker="AAPL")
    ev2 = to_evidence(result2, "sq_propagation2")
    print(ev2.model_dump())

    print("\n--- no target specified, general 'who's affected' ---")
    result3 = find_paths(StubGraph(), "Input", "graphite", "graphite")
    ev3 = to_evidence(result3, "sq_propagation3")
    print(ev3.model_dump())