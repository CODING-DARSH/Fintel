# =============================================================================
# src/agents/detect_contradictions.py
# =============================================================================
# STEP 6 of the agent build: contradiction detection.
#
# Deliberately narrow scope for this first pass:
#   - Only compares Evidence pairs sharing the same primary_entity.
#     Cross-company evidence isn't a contradiction, it's just two
#     different companies — comparing it would be noise, not signal.
#   - One LLM call PER PAIR, not one big call over all evidence at once.
#     Keeps prompts small and each judgment independently checkable.
#     Trade-off: more LLM calls than a single-shot approach — acceptable
#     for now given evidence lists are still small (k=5 per sub-question).
#     Revisit if pair count grows (see _pairs_for_entity note below).
#   - Skips pairs that share the exact same source (same chunk_id, or
#     same graph node) — can't contradict itself.
#   - Same llm_call injection pattern as decompose.py:
#         call_llm(prompt: str) -> Optional[dict]
#     If a call fails, that PAIR is skipped, not silently marked
#     "no contradiction" — a failed check and a checked-and-clear result
#     are different things and shouldn't collapse into each other.
# =============================================================================

from __future__ import annotations

import logging
from itertools import combinations
from typing import Callable, Optional

from src.agents.models.contract import Evidence, Contradiction

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]

# Hard ceiling on pairs actually sent to the LLM per entity. Lowered from
# an initial 30 to 12 after a real run showed 41 candidate pairs for a
# single entity in just one hop (3 sub-questions, k=5) — 30 sequential
# LLM calls for contradiction-checking alone was genuinely too slow.
# This is a stopgap ceiling, not a fix for the root cause — see
# run_reasoning_loop.py's evidence caching, which is the real fix
# (avoids re-generating near-duplicate evidence across hops in the
# first place, so pair count shouldn't realistically hit this cap
# once that's in place).
MAX_PAIRS_PER_ENTITY = 6


CONTRADICTION_PROMPT_TEMPLATE = """You are checking two pieces of financial evidence about the
same company for a genuine factual contradiction (not just different
topics, not just different time periods describing normal change).

Evidence A ({date_a}): {text_a}

Evidence B ({date_b}): {text_b}

Does B genuinely contradict A (not just update it, not just discuss a
different aspect)? A real contradiction means the two claims cannot
both be true of the same point in time — e.g. one flatly denies what
the other asserts. A company's cost situation changing over time is
NOT a contradiction; it's just change.

Return ONLY valid JSON, no preamble, no markdown fences:

{{
  "is_contradiction": true or false,
  "description": "one sentence describing the conflict, or empty string if none",
  "resolution": "one sentence on how to reconcile them, or empty string if not a contradiction"
}}
"""


def _group_by_entity(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    groups: dict[str, list[Evidence]] = {}
    for e in evidence:
        groups.setdefault(e.primary_entity, []).append(e)
    return groups


def _same_source(a: Evidence, b: Evidence) -> bool:
    if a.source.type != b.source.type:
        return False
    if a.source.type == "vector":
        return a.source.chunk_id == b.source.chunk_id and a.source.chunk_id is not None
    # graph: treat as same source if both point at the same node/result_type
    # with overlapping source chunks
    return (
        a.source.result_type == b.source.result_type
        and set(a.source.chunks) == set(b.source.chunks)
        and a.source.result_type is not None
    )


def _pairs_for_entity(evidence: list[Evidence]) -> list[tuple[Evidence, Evidence]]:
    pairs = [
        (a, b) for a, b in combinations(evidence, 2)
        if not _same_source(a, b)
    ]
    if len(pairs) > MAX_PAIRS_PER_ENTITY:
        log.warning(
            f"Entity has {len(pairs)} candidate pairs, capping at "
            f"{MAX_PAIRS_PER_ENTITY} — evidence set may be too large "
            f"for pairwise contradiction checking at this scale."
        )
        pairs = pairs[:MAX_PAIRS_PER_ENTITY]
    return pairs


def _check_pair(a: Evidence, b: Evidence, llm_call: LlmCallFn) -> Optional[Contradiction]:
    prompt = CONTRADICTION_PROMPT_TEMPLATE.format(
        date_a=a.recency, text_a=a.text,
        date_b=b.recency, text_b=b.text,
    )

    try:
        raw = llm_call(prompt)
    except Exception as e:
        log.warning(f"contradiction check raised for pair ({a.id}, {b.id}): {e}")
        return None

    if not raw or not isinstance(raw, dict):
        log.warning(f"contradiction check returned unparseable result for pair ({a.id}, {b.id})")
        return None

    if not raw.get("is_contradiction"):
        return None

    return Contradiction(
        evidence_ids=[a.id, b.id],
        description=raw.get("description", "").strip() or "Contradiction flagged, no description given",
        resolution=(raw.get("resolution") or "").strip() or None,
    )


def detect_contradictions(
    evidence: list[Evidence],
    llm_call: LlmCallFn,
    already_checked: Optional[set] = None,
) -> tuple[list[Contradiction], set]:
    """
    Scan evidence for contradictions, scoped to pairs sharing the same
    primary_entity. Returns only pairs the model actually flagged —
    failed/unparseable checks are skipped, not treated as "no contradiction".

    already_checked: a set of frozenset({evidence_id_a, evidence_id_b})
    pairs already checked in a previous call (e.g. an earlier hop in
    run_reasoning_loop.py). Pairs already in this set are skipped
    entirely — no re-checking evidence that hasn't changed just because
    the loop ran again. Pass None (default) for a fresh, uncached scan.

    Returns (contradictions_found, updated_checked_set) — the caller is
    expected to carry the returned set into the next call, so checking
    stays cumulative across hops rather than resetting each time.
    """
    already_checked = already_checked if already_checked is not None else set()
    contradictions: list[Contradiction] = []
    groups = _group_by_entity(evidence)

    for entity, ev_list in groups.items():
        if len(ev_list) < 2:
            continue
        for a, b in _pairs_for_entity(ev_list):
            pair_key = frozenset((a.id, b.id))
            if pair_key in already_checked:
                continue
            already_checked.add(pair_key)
            result = _check_pair(a, b, llm_call)
            if result:
                contradictions.append(result)

    return contradictions, already_checked


if __name__ == "__main__":
    # Manual smoke test with a fake llm_call — no real API hit.
    from src.agents.models.contract import SourceRef

    def fake_llm_call(prompt: str) -> Optional[dict]:
        # crude: flag as contradiction if both texts are in the prompt
        # and one contains "not" and the other doesn't — just enough to
        # exercise both branches without a real model
        if "not mentioned as direct risk" in prompt and "blamed on material costs" in prompt:
            return {
                "is_contradiction": True,
                "description": "2021 filing says steel isn't a direct risk, but 2022 8-K blames logistics costs on material costs",
                "resolution": "Exposure is indirect/downstream, not a named risk factor",
            }
        return {"is_contradiction": False, "description": "", "resolution": ""}

    ev1 = Evidence(
        id="ev1", sub_question_id="sq1",
        source=SourceRef(type="vector", chunk_id="c1", filing_id="f2021"),
        text="AMZN 2021 10-K: steel not mentioned as direct risk factor.",
        primary_entity="AMZN", reliability=1.0, recency="2021-01-01",
        relevance=0.8, confidence=0.7,
    )
    ev2 = Evidence(
        id="ev2", sub_question_id="sq1",
        source=SourceRef(type="vector", chunk_id="c2", filing_id="f2022"),
        text="AMZN 2022 8-K: logistics cost increase blamed on material costs.",
        primary_entity="AMZN", reliability=1.0, recency="2022-06-01",
        relevance=0.8, confidence=0.7,
    )
    ev3 = Evidence(
        id="ev3", sub_question_id="sq2",
        source=SourceRef(type="graph", result_type="macro_impact", chunks=["c99"]),
        text="Steel price +12% this month.",
        primary_entity="AMZN", reliability=0.9, recency="current",
        relevance=0.9, confidence=0.8,
    )
    ev_other_entity = Evidence(
        id="ev4", sub_question_id="sq3",
        source=SourceRef(type="vector", chunk_id="c3", filing_id="f2022b"),
        text="TSLA has no steel-related disclosures.",
        primary_entity="TSLA", reliability=1.0, recency="2022-01-01",
        relevance=0.5, confidence=0.5,
    )

    result, checked = detect_contradictions([ev1, ev2, ev3, ev_other_entity], fake_llm_call)
    print(f"Found {len(result)} contradiction(s), {len(checked)} pair(s) checked")
    for c in result:
        print(c.model_dump())

    print("\n--- second call with same evidence + already_checked passed in ---")
    print("--- should find 0 NEW contradictions and check 0 NEW pairs ---")
    result2, checked2 = detect_contradictions(
        [ev1, ev2, ev3, ev_other_entity], fake_llm_call, already_checked=checked,
    )
    print(f"Found {len(result2)} new contradiction(s), {len(checked2)} total pair(s) checked")