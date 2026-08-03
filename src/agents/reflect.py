# =============================================================================
# src/agents/reflect.py
# =============================================================================
# STEP 9a of the agent build: the reflection DECISION, isolated from the
# loop that acts on it (that's run_reasoning_loop.py, step 9b).
#
# Given the current state after a hop (hypothesis, gaps, contradictions),
# decide: is this good enough, or do we need another hop? If another hop
# is needed, produce ONE new, sharper SubQuestion targeting specifically
# what's missing — not a re-run of the original decomposition.
#
# This is the piece that actually makes the system multi-hop rather than
# "decompose once, retrieve once, answer." Everything in steps 2-7 runs
# INSIDE one hop; this decides whether to take another one.
# =============================================================================

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.agents.models.contract import Gap, Contradiction, Hypothesis, SubQuestion

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]

# Below this, the hypothesis isn't trusted enough to stop on confidence
# alone — combined with "no gaps left" is what actually allows stopping.
# This is a floor, not the only stopping signal (see reflect() below).
MIN_STOP_CONFIDENCE = 0.6


REFLECT_PROMPT_TEMPLATE = """You are deciding whether a financial research answer is ready, or
needs one more round of targeted research.

Original question: {query}

Current hypothesis: {hypothesis_statement}
Current confidence: {confidence}

Unresolved gaps:
{gaps_block}

Unresolved contradictions:
{contradictions_block}

Decide: does this need ONE more round of research, targeting specifically
what's missing? Don't ask for another round just to be thorough — only if
a gap or contradiction is actually likely to change the answer in a
meaningful way. If the gaps are minor or unlikely to change the
conclusion, say so is not needed.

If another round IS needed, write ONE new, sharper sub-question that
targets the single most important gap or contradiction — not a repeat of
the original question.

Return ONLY valid JSON, no preamble, no markdown fences:

{{
  "needs_another_round": true or false,
  "reason": "one sentence explaining the decision either way",
  "new_sub_question": "the sharper follow-up question, or empty string if not needed",
  "new_sub_question_focus": "short lowercase label for what kind of evidence it needs, or empty string"
}}
"""


def _format_gaps_block(gaps: list[Gap]) -> str:
    if not gaps:
        return "(none)"
    return "\n".join(f"  - {g.description}" for g in gaps)


def _format_contradictions_block(contradictions: list[Contradiction]) -> str:
    unresolved = [c for c in contradictions if not c.resolution]
    if not unresolved:
        return "(none)"
    return "\n".join(f"  - {c.description}" for c in unresolved)


class ReflectionDecision:
    """Plain result object — not part of the main contract since it's
    internal to the loop, never returned to a caller outside agents/."""
    def __init__(self, needs_another_round: bool, reason: str,
                 new_sub_question: Optional[SubQuestion] = None):
        self.needs_another_round = needs_another_round
        self.reason = reason
        self.new_sub_question = new_sub_question


def reflect(
    query: str,
    hop_number: int,
    hypothesis: Hypothesis,
    gaps: list[Gap],
    contradictions: list[Contradiction],
    llm_call: LlmCallFn,
) -> ReflectionDecision:
    """
    Decide whether another hop is needed. Deterministic short-circuits
    happen BEFORE any LLM call — no reason to spend a call asking "should
    we continue" when the answer is already obvious from the numbers:
      - no gaps AND no unresolved contradictions AND confidence is
        already high -> stop, no LLM call needed.
      - hop_number already at a sane hard ceiling handled by the CALLER
        (run_reasoning_loop respects max_reflection_loops itself) — this
        function doesn't enforce that cap, it only ever answers "would
        one more hop help", the caller decides whether one is allowed.
    """
    unresolved_contradictions = [c for c in contradictions if not c.resolution]

    if not gaps and not unresolved_contradictions and hypothesis.confidence >= MIN_STOP_CONFIDENCE:
        return ReflectionDecision(
            needs_another_round=False,
            reason=(
                f"No gaps or unresolved contradictions, confidence "
                f"{hypothesis.confidence:.2f} >= {MIN_STOP_CONFIDENCE} — stopping."
            ),
        )

    prompt = REFLECT_PROMPT_TEMPLATE.format(
        query=query,
        hypothesis_statement=hypothesis.statement,
        confidence=f"{hypothesis.confidence:.2f}",
        gaps_block=_format_gaps_block(gaps),
        contradictions_block=_format_contradictions_block(contradictions),
    )

    try:
        raw = llm_call(prompt)
    except Exception as e:
        log.warning(f"reflect() llm_call raised: {e} — defaulting to stop, not looping blind")
        return ReflectionDecision(
            needs_another_round=False,
            reason=f"Reflection check failed ({e}) — stopping rather than looping without judgment.",
        )

    if not raw or not isinstance(raw, dict):
        log.warning("reflect() got unparseable result — defaulting to stop")
        return ReflectionDecision(
            needs_another_round=False,
            reason="Reflection check returned unparseable result — stopping rather than looping blind.",
        )

    if not raw.get("needs_another_round"):
        return ReflectionDecision(
            needs_another_round=False,
            reason=raw.get("reason", "Model determined no further research needed."),
        )

    new_q_text = (raw.get("new_sub_question") or "").strip()
    if not new_q_text:
        # Model said yes but gave nothing to act on — can't loop on nothing.
        log.warning("reflect() said needs_another_round=True but gave no new_sub_question — stopping")
        return ReflectionDecision(
            needs_another_round=False,
            reason="Model flagged more research needed but did not provide a usable follow-up question.",
        )

    focus = (raw.get("new_sub_question_focus") or "").strip().lower() or "filing"

    return ReflectionDecision(
        needs_another_round=True,
        reason=raw.get("reason", "Model determined further research needed."),
        new_sub_question=SubQuestion(
            id=f"reflect_h{hop_number}",
            text=new_q_text,
            retrieval_focus=focus,
        ),
    )


if __name__ == "__main__":
    from src.agents.models.contract import Hypothesis

    print("--- deterministic stop: no gaps, no contradictions, high confidence ---")
    d = reflect(
        "test query", 1,
        Hypothesis(statement="x", supporting_evidence_ids=[], confidence=0.8, limiting_factors=[]),
        [], [], lambda p: None,  # llm_call never invoked here — should short-circuit
    )
    print(d.needs_another_round, "|", d.reason)

    print("\n--- gap present, model says continue with a sharper question ---")
    def fake_llm_call(prompt):
        return {
            "needs_another_round": True,
            "reason": "Hedging data is directly relevant to the confidence of the conclusion.",
            "new_sub_question": "Does AMZN disclose any commodity hedging instruments in its 10-K?",
            "new_sub_question_focus": "filing",
        }
    d = reflect(
        "Should I be worried about AMZN's steel exposure?", 1,
        Hypothesis(statement="x", supporting_evidence_ids=[], confidence=0.65, limiting_factors=[]),
        [Gap(sub_question_id="sq3", description="No hedging data found")], [], fake_llm_call,
    )
    print(d.needs_another_round, "|", d.reason)
    print(d.new_sub_question.model_dump() if d.new_sub_question else None)

    print("\n--- model says continue but gives no question -> forced stop ---")
    d = reflect(
        "test query", 1,
        Hypothesis(statement="x", supporting_evidence_ids=[], confidence=0.5, limiting_factors=[]),
        [Gap(sub_question_id="sq1", description="something missing")], [],
        lambda p: {"needs_another_round": True, "reason": "need more", "new_sub_question": ""},
    )
    print(d.needs_another_round, "|", d.reason)

    print("\n--- broken llm_call -> forced stop, not blind loop ---")
    d = reflect(
        "test query", 1,
        Hypothesis(statement="x", supporting_evidence_ids=[], confidence=0.5, limiting_factors=[]),
        [Gap(sub_question_id="sq1", description="something missing")], [],
        lambda p: (_ for _ in ()).throw(RuntimeError("API down")),
    )
    print(d.needs_another_round, "|", d.reason)