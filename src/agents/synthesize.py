# =============================================================================
# src/agents/synthesize.py
# =============================================================================
# STEP 8 of the agent build: synthesis.
#
# Takes the fully-reasoned state from run_reasoning_loop() (evidence,
# gaps, contradictions, hypothesis) and produces the final prose fields:
#   answer, reasoning_chain, citations, follow_up_questions
#
# Runs ONCE, on whatever the reasoning loop settled on — not per hop.
# Same llm_call injection pattern as every other step.
#
# Design choices:
#   - citations are built from supporting_evidence_ids ONLY (what the
#     hypothesis actually leaned on), not all evidence gathered — an
#     answer shouldn't cite evidence it didn't use just because it was
#     retrieved.
#   - follow_up_questions has a deterministic fallback path: if the LLM
#     call fails, we still produce follow-ups by directly reframing
#     unresolved gaps as questions, rather than returning nothing.
#   - if hypothesis itself is the empty-evidence/failure fallback
#     (confidence 0.0, no supporting evidence), synthesis still runs but
#     the prompt is built to make that limitation explicit in the answer
#     rather than let the model paper over it with confident prose.
# =============================================================================

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.agents.models.contract import (
    Evidence, Gap, Contradiction, Hypothesis, SourceRef,
)

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]


class SynthesisResult:
    def __init__(self, answer: str, reasoning_chain: str,
                 citations: list[SourceRef], follow_up_questions: list[str]):
        self.answer = answer
        self.reasoning_chain = reasoning_chain
        self.citations = citations
        self.follow_up_questions = follow_up_questions


SYNTHESIS_PROMPT_TEMPLATE = """You are writing the final answer to a financial research question,
based on a hypothesis already formed from evidence.

Original question: {query}

Hypothesis (confidence {confidence}): {hypothesis_statement}

Limiting factors on this confidence:
{limiting_factors_block}

Unresolved contradictions:
{contradictions_block}

Write the final answer. Requirements:
- Be direct — state the conclusion, don't hedge more than the confidence
  score warrants, but don't overstate certainty either.
- If confidence is low or evidence is missing, say so plainly in the
  answer itself — do not write confidently past what the hypothesis
  supports.
- Include a short reasoning chain showing how the evidence leads to the
  conclusion (e.g. "steel prices up -> capex pressure -> margin impact").
- Suggest 1-3 follow-up questions a person might reasonably ask next,
  informed by the limiting factors above where relevant.

Return ONLY valid JSON, no preamble, no markdown fences:

{{
  "answer": "the final answer, 2-5 sentences",
  "reasoning_chain": "short A -> B -> C style chain",
  "follow_up_questions": ["question 1", "question 2"]
}}
"""


def _format_limiting_factors(factors: list[str]) -> str:
    if not factors:
        return "(none)"
    return "\n".join(f"  - {f}" for f in factors)


def _format_contradictions(contradictions: list[Contradiction]) -> str:
    unresolved = [c for c in contradictions if not c.resolution]
    if not unresolved:
        return "(none)"
    return "\n".join(f"  - {c.description}" for c in unresolved)


def _build_citations(evidence: list[Evidence], supporting_ids: list[str]) -> list[SourceRef]:
    by_id = {e.id: e for e in evidence}
    citations = []
    for eid in supporting_ids:
        ev = by_id.get(eid)
        if ev:
            citations.append(ev.source)
    return citations


def _fallback_follow_ups(gaps: list[Gap]) -> list[str]:
    """Deterministic fallback: reframe unresolved gaps as questions
    directly, rather than returning nothing if the LLM call fails."""
    questions = []
    for g in gaps[:3]:
        questions.append(f"Can we find more on: {g.description}?")
    return questions


def _fallback_synthesis(
    hypothesis: Hypothesis, evidence: list[Evidence], gaps: list[Gap],
) -> SynthesisResult:
    """
    Used if the LLM call fails or returns something unparseable. Falls
    back to rendering the hypothesis statement directly rather than
    fabricating prose — the hypothesis itself already went through its
    own fallback discipline (form_hypothesis.py), so this is a safe
    thing to surface as-is.
    """
    answer = hypothesis.statement
    if hypothesis.limiting_factors:
        answer += " Limiting factors: " + "; ".join(hypothesis.limiting_factors) + "."
    return SynthesisResult(
        answer=answer,
        reasoning_chain="(synthesis unavailable — showing raw hypothesis)",
        citations=_build_citations(evidence, hypothesis.supporting_evidence_ids),
        follow_up_questions=_fallback_follow_ups(gaps),
    )


def synthesize(
    query: str,
    evidence: list[Evidence],
    gaps: list[Gap],
    contradictions: list[Contradiction],
    hypothesis: Hypothesis,
    llm_call: LlmCallFn,
) -> SynthesisResult:
    """
    Produce the final answer/reasoning_chain/citations/follow_up_questions
    from the reasoning loop's settled state. Runs once.
    """
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        query=query,
        confidence=f"{hypothesis.confidence:.2f}",
        hypothesis_statement=hypothesis.statement,
        limiting_factors_block=_format_limiting_factors(hypothesis.limiting_factors),
        contradictions_block=_format_contradictions(contradictions),
    )

    try:
        raw = llm_call(prompt)
    except Exception as e:
        log.error(f"llm_call raised during synthesis: {e}")
        return _fallback_synthesis(hypothesis, evidence, gaps)

    if not raw or not isinstance(raw, dict) or "answer" not in raw:
        log.warning("Synthesis returned unparseable result — using fallback")
        return _fallback_synthesis(hypothesis, evidence, gaps)

    follow_ups = raw.get("follow_up_questions") or []
    if not isinstance(follow_ups, list) or not follow_ups:
        follow_ups = _fallback_follow_ups(gaps)

    return SynthesisResult(
        answer=raw["answer"],
        reasoning_chain=raw.get("reasoning_chain", ""),
        citations=_build_citations(evidence, hypothesis.supporting_evidence_ids),
        follow_up_questions=follow_ups,
    )


if __name__ == "__main__":
    from src.agents.models.contract import SourceRef

    ev1 = Evidence(
        id="ev1", sub_question_id="sq1",
        source=SourceRef(type="vector", chunk_id="c1", filing_id="f1"),
        text="AMZN discusses logistics equipment costs.",
        primary_entity="AMZN", reliability=1.0, recency="2022-01-01",
        relevance=0.8, confidence=0.75,
    )
    hyp = Hypothesis(
        statement=(
            "AMZN has indirect steel exposure via logistics/fulfillment capex, "
            "not primary COGS."
        ),
        supporting_evidence_ids=["ev1"],
        confidence=0.72,
        limiting_factors=["no hedging data found"],
    )
    gaps = [Gap(sub_question_id="sq3", description="No hedging data found for AMZN steel exposure")]

    def fake_llm_call(prompt: str) -> Optional[dict]:
        return {
            "answer": (
                "AMZN's steel exposure is indirect, flowing through fulfillment "
                "center capex rather than direct input costs. Given current price "
                "trends, expect modest margin pressure over 12-18 months. "
                "Confidence is moderate — no hedging data was found."
            ),
            "reasoning_chain": "steel price up -> capex cost up -> margin pressure lagged by depreciation cycle",
            "follow_up_questions": [
                "Want me to check AMZN's capex trend over the last 4 quarters?",
                "Want current steel futures pricing for a longer-term view?",
            ],
        }

    result = synthesize("Should I worry about AMZN steel exposure?", [ev1], gaps, [], hyp, fake_llm_call)
    print("ANSWER:", result.answer)
    print("REASONING CHAIN:", result.reasoning_chain)
    print("CITATIONS:", [c.model_dump() for c in result.citations])
    print("FOLLOW-UPS:", result.follow_up_questions)

    print("\n--- fallback path (broken llm_call) ---")
    result2 = synthesize("Should I worry about AMZN steel exposure?", [ev1], gaps, [], hyp, lambda p: None)
    print("ANSWER:", result2.answer)
    print("REASONING CHAIN:", result2.reasoning_chain)
    print("FOLLOW-UPS:", result2.follow_up_questions)