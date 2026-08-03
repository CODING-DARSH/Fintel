# =============================================================================
# src/agents/run_reasoning_loop.py
# =============================================================================
# STEP 9b of the agent build: the actual multi-hop loop.
#
# Wires together everything built in steps 2-7, plus reflect() (9a), into
# a real loop:
#
#   decompose_query (once)
#     -> gather_all_evidence -> detect_gaps -> detect_contradictions
#        -> form_hypothesis -> reflect
#            -> if continue: ONE new sub-question added, loop again
#               (re-gathers evidence for ALL sub-questions so far —
#                see note in run_reasoning_loop() on why)
#            -> if stop: done
#
# This does NOT include synthesis (step 8) — this loop's output is the
# fully-reasoned state (sub_questions, evidence, contradictions, gaps,
# hypothesis, reflection_count, stopped_reason). Turning that into final
# prose (answer/reasoning_chain/citations/follow_up_questions) is a
# separate, later step precisely because synthesis should run ONCE on
# whatever this loop settles on, not get re-run every hop.
# =============================================================================

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.agents.models.contract import (
    SubQuestion, Evidence, Gap, Contradiction, Hypothesis,
)
from src.agents.decompose import decompose_query
from src.agents.gather_all_evidence import gather_all_evidence, detect_gaps
from src.agents.detect_contradictions import detect_contradictions
from src.agents.form_hypothesis import form_hypothesis
from src.agents.reflect import reflect

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]


class ReasoningLoopResult:
    """
    Internal result of the loop — everything OrchestratorResponse needs
    EXCEPT the synthesis fields (answer, reasoning_chain, citations,
    follow_up_questions), which get filled in by the synthesis step.
    """
    def __init__(
        self,
        sub_questions: list[SubQuestion],
        evidence: list[Evidence],
        gaps: list[Gap],
        contradictions: list[Contradiction],
        hypothesis: Hypothesis,
        reflection_count: int,
        stopped_reason: str,
    ):
        self.sub_questions = sub_questions
        self.evidence = evidence
        self.gaps = gaps
        self.contradictions = contradictions
        self.hypothesis = hypothesis
        self.reflection_count = reflection_count
        self.stopped_reason = stopped_reason


def run_reasoning_loop(
    query: str,
    llm_call: LlmCallFn,
    max_reflection_loops: int = 2,
    k_per_hop: int = 5,
) -> ReasoningLoopResult:
    """
    Run the full multi-hop reasoning loop for one query.

    NOTE on re-gathering evidence each hop: when reflect() adds a new
    sub-question, we re-run gather_all_evidence() across ALL
    sub-questions so far, not just the new one. This is deliberate, not
    wasteful-by-accident: gap detection and contradiction detection need
    to see the FULL evidence set to make correct calls (e.g. a
    contradiction between hop-1 evidence and hop-2 evidence would be
    missed if hop-2 only re-checked its own new evidence against itself).
    NOTE on evidence caching: gather_all_evidence() and detect_contradictions()
    both accept caches (evidence_cache, already_checked) that persist across
    the while loop below. This means when reflect() adds a new sub-question,
    only THAT sub-question gets newly retrieved, and only NEW evidence pairs
    get contradiction-checked — hop 1's sub-questions are not re-retrieved
    or re-checked on every subsequent hop. Fixed after a real run showed
    this mattering in practice (41 candidate pairs from just 3 sub-questions
    across 1 reflection hop, each requiring a sequential LLM call).
    """
    sub_questions = decompose_query(query, llm_call)
    reflection_count = 0
    stopped_reason = "not_run"

    evidence: list[Evidence] = []
    gaps: list[Gap] = []
    contradictions: list[Contradiction] = []
    hypothesis: Optional[Hypothesis] = None

    evidence_cache: dict[str, list[Evidence]] = {}
    checked_pairs: set = set()

    while True:
        evidence = gather_all_evidence(sub_questions, k=k_per_hop, evidence_cache=evidence_cache)
        gaps = detect_gaps(sub_questions, evidence)

        new_contradictions, checked_pairs = detect_contradictions(
            evidence, llm_call, already_checked=checked_pairs,
        )
        contradictions = contradictions + new_contradictions

        hypothesis = form_hypothesis(query, evidence, gaps, contradictions, llm_call)

        if reflection_count >= max_reflection_loops:
            stopped_reason = "max_reflections"
            log.info(f"Stopping: hit max_reflection_loops ({max_reflection_loops})")
            break

        decision = reflect(
            query, reflection_count + 1, hypothesis, gaps, contradictions, llm_call,
        )

        if not decision.needs_another_round:
            stopped_reason = "synthesized"
            log.info(f"Stopping: {decision.reason}")
            break

        new_sq = decision.new_sub_question
        already_asked = any(sq.text == new_sq.text for sq in sub_questions)
        if already_asked:
            # reflect() proposed a question we've effectively already asked —
            # treat as no new information available, stop rather than loop
            # on a duplicate.
            stopped_reason = "no_new_evidence"
            log.info("Stopping: reflect() proposed a duplicate sub-question")
            break

        sub_questions = sub_questions + [new_sq]
        reflection_count += 1
        log.info(f"Hop {reflection_count}: {decision.reason} -> new sub-question: {new_sq.text!r}")

    return ReasoningLoopResult(
        sub_questions=sub_questions,
        evidence=evidence,
        gaps=gaps,
        contradictions=contradictions,
        hypothesis=hypothesis,
        reflection_count=reflection_count,
        stopped_reason=stopped_reason,
    )


if __name__ == "__main__":
    # Structural smoke test — fake llm_call plays every role (decompose,
    # contradiction check, hypothesis, reflect) so this runs without any
    # real API or live Neo4j/Chroma reachable. gather_all_evidence will
    # still hit the live retrieval stack and come back empty in this
    # sandbox (no DBs running) — that's fine, it exercises the loop's
    # control flow including the empty-evidence hypothesis fallback.

    call_count = {"n": 0}

    def fake_llm_call(prompt: str) -> Optional[dict]:
        call_count["n"] += 1
        if "decomposing a financial research question" in prompt:
            return {"sub_questions": [
                {"text": "Does AMZN have steel exposure?", "retrieval_focus": "filing"},
            ]}
        if "checking two pieces of financial evidence" in prompt:
            return {"is_contradiction": False, "description": "", "resolution": ""}
        if "forming a single structured hypothesis" in prompt:
            return {
                "statement": "Insufficient evidence to form a strong hypothesis.",
                "supporting_evidence_ids": [],
                "confidence": 0.3,
                "limiting_factors": ["no evidence retrieved in this environment"],
            }
        if "deciding whether a financial research answer is ready" in prompt:
            # only allow ONE hop in this smoke test, then force stop
            if call_count["n"] > 10:
                return {"needs_another_round": False, "reason": "smoke test cap"}
            return {
                "needs_another_round": True,
                "reason": "confidence too low, evidence is empty in this sandbox",
                "new_sub_question": "What is the current steel price trend?",
                "new_sub_question_focus": "macro",
            }
        return None

    result = run_reasoning_loop(
        "Should I be worried about AMZN's steel exposure given current commodity prices?",
        fake_llm_call,
        max_reflection_loops=2,
        k_per_hop=3,
    )

    print(f"stopped_reason     : {result.stopped_reason}")
    print(f"reflection_count   : {result.reflection_count}")
    print(f"sub_questions      : {[sq.text for sq in result.sub_questions]}")
    print(f"evidence count     : {len(result.evidence)}")
    print(f"gaps               : {[g.description for g in result.gaps]}")
    print(f"hypothesis         : {result.hypothesis.model_dump()}")