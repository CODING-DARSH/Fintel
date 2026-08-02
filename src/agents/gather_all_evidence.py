# =============================================================================
# src/agents/gather_all_evidence.py
# =============================================================================
# STEP 4 + 5 of the agent build: evidence across ALL sub-questions, then
# gap detection. No LLM call in this file — both steps are pure logic on
# data already produced by decompose_query() + gather_evidence().
#
# STEP 4 — gather_all_evidence():
#   Loop gather_evidence() over every SubQuestion, collect into one
#   list[Evidence]. Retriever/merger/parser instances are created once
#   and shared across sub-questions (not reconnected per call).
#
# STEP 5 — detect_gaps():
#   For each SubQuestion, check whether it got zero evidence, or only
#   weak/low-relevance evidence, and emit a Gap. Threshold-based for now
#   (documented below) — no LLM judgment yet. That's a deliberate scope
#   limit, not an oversight: "did anything come back, and was it any
#   good by the numbers we already have" is answerable without a model
#   call, so it doesn't need one yet. An LLM-judged version ("does this
#   evidence actually answer the sub-question semantically") is a
#   plausible future upgrade to this same function, not a different one.
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional

from src.retrieval import GraphRetriever, VectorRetriever, RetrievalMerger, QueryParser
from src.agents.models.contract import SubQuestion, Evidence, Gap
from src.agents.gather_evidence import gather_evidence

log = logging.getLogger(__name__)

# Below this relevance score, evidence counts as "weak" rather than
# "found" for gap-detection purposes. Not the same threshold as
# retrieval's own scoring — this is specifically about whether a
# sub-question should be considered answered.
WEAK_RELEVANCE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

def gather_all_evidence(
    sub_questions: list[SubQuestion],
    k: int = 5,
    evidence_cache: Optional[dict[str, list[Evidence]]] = None,
) -> list[Evidence]:
    """
    Run gather_evidence() for every sub-question, using ONE shared set of
    retriever/merger/parser instances (created here, not per sub-question)
    so a multi-sub-question pass doesn't reconnect to Neo4j/Chroma
    repeatedly.

    evidence_cache: optional dict of sub_question.id -> list[Evidence],
    mutated in place. If provided, sub-questions whose id is already a
    key in the cache are SKIPPED entirely (no re-retrieval) and their
    cached evidence is reused as-is. This is what makes run_reasoning_loop's
    reflection hops cheap — hop 2 only retrieves for the ONE new
    sub-question reflect() added, not all of them again.

    Pass evidence_cache=None (default) for the old behavior: always
    retrieve everything fresh, no caching.
    """
    graph  = GraphRetriever()
    vector = VectorRetriever()
    merger = RetrievalMerger()
    parser = QueryParser()

    # Shared across ALL sub-questions in this call (i.e. within one hop)
    # so multiple sub-questions resolving to the same propagation start
    # entity (e.g. several all anchoring on "china") reuse one traversal
    # instead of each re-running it independently.
    propagation_cache: dict = {}

    all_evidence: list[Evidence] = []
    for sq in sub_questions:
        if evidence_cache is not None and sq.id in evidence_cache:
            all_evidence.extend(evidence_cache[sq.id])
            continue

        ev = gather_evidence(
            sq, k=k,
            graph=graph, vector=vector, merger=merger, parser=parser,
            propagation_cache=propagation_cache,
        )
        if evidence_cache is not None:
            evidence_cache[sq.id] = ev
        all_evidence.extend(ev)

    return all_evidence


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------

def detect_gaps(
    sub_questions: list[SubQuestion],
    evidence: list[Evidence],
) -> list[Gap]:
    """
    For each sub-question, check whether it has any evidence at all, and
    whether that evidence clears WEAK_RELEVANCE_THRESHOLD. Emits a Gap
    per sub-question that fails either check — never silently drops a
    sub-question that went unanswered.
    """
    by_sub_question: dict[str, list[Evidence]] = {}
    for ev in evidence:
        by_sub_question.setdefault(ev.sub_question_id, []).append(ev)

    gaps: list[Gap] = []
    for sq in sub_questions:
        sq_evidence = by_sub_question.get(sq.id, [])

        if not sq_evidence:
            gaps.append(Gap(
                sub_question_id=sq.id,
                description=f"No evidence found for: {sq.text!r}",
            ))
            continue

        strong = [e for e in sq_evidence if e.relevance >= WEAK_RELEVANCE_THRESHOLD]
        if not strong:
            best = max(sq_evidence, key=lambda e: e.relevance)
            gaps.append(Gap(
                sub_question_id=sq.id,
                description=(
                    f"Only weak evidence found for: {sq.text!r} "
                    f"(best relevance {best.relevance:.2f}, "
                    f"{len(sq_evidence)} item(s) considered)"
                ),
            ))

    return gaps


if __name__ == "__main__":
    # Structural smoke test with fake Evidence — no DB needed here since
    # detect_gaps() is pure logic. gather_all_evidence() still needs live
    # Neo4j/Chroma, same as gather_evidence.py's own smoke test.
    sub_questions = [
        SubQuestion(id="sq1", text="Does AMZN have steel exposure?", retrieval_focus="filing"),
        SubQuestion(id="sq2", text="What is current steel price trend?", retrieval_focus="macro"),
        SubQuestion(id="sq3", text="What is AMZN's hedging strategy for steel?", retrieval_focus="filing"),
    ]

    fake_evidence = [
        Evidence(
            id="sq1_ev1", sub_question_id="sq1",
            source={"type": "vector", "chunk_id": "c1", "filing_id": "f1"},
            text="AMZN discusses logistics equipment costs...",
            primary_entity="AMZN", reliability=1.0, recency="2022-10-01",
            relevance=0.85, confidence=0.8,
        ),
        Evidence(
            id="sq2_ev1", sub_question_id="sq2",
            source={"type": "graph", "result_type": "macro_impact", "chunks": []},
            text="Steel price signal, weak match to query",
            primary_entity="AMZN", reliability=0.9, recency="current",
            relevance=0.2, confidence=0.3,
        ),
        # sq3 gets nothing — should surface as a real gap
    ]

    gaps = detect_gaps(sub_questions, fake_evidence)
    for g in gaps:
        print(g.model_dump())