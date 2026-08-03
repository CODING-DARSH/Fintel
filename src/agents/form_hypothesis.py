# =============================================================================
# src/agents/form_hypothesis.py
# =============================================================================
# STEP 7 of the agent build: hypothesis formation.
#
# Takes everything gathered so far — evidence, gaps, contradictions — and
# produces ONE Hypothesis: a statement, a confidence score, and explicit
# limiting factors. This is deliberately NOT the final answer (that's
# synthesis, step 8) — it's the structured "what do we actually believe,
# and how sure are we" step that synthesis will later turn into prose.
#
# Same llm_call injection pattern as decompose.py / detect_contradictions.py:
#       call_llm(prompt: str) -> Optional[dict]
#
# Design choices worth flagging:
#   - Confidence is model-reported, not computed from a formula. We DO
#     feed it the raw ingredients (evidence count/reliability, gap count,
#     contradiction count) so its number is grounded in something
#     concrete rather than vibes, but the arithmetic of "how much do
#     gaps/contradictions lower confidence" is left to the model's
#     judgment for now. A deterministic confidence formula is a
#     reasonable future upgrade, not built here to avoid guessing at
#     weights with no real query volume to calibrate against yet.
#   - limiting_factors are drawn directly from gap descriptions and
#     contradiction descriptions — not re-invented by the model. This
#     keeps the hypothesis honest: it can't claim high confidence while
#     silently ignoring a gap that's already been detected.
#   - If evidence is empty entirely, we don't call the LLM at all — an
#     empty-evidence hypothesis is a known, deterministic case ("nothing
#     found"), not something worth spending a call on.
# =============================================================================

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.agents.models.contract import Evidence, Gap, Contradiction, Hypothesis

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]


HYPOTHESIS_PROMPT_TEMPLATE = """You are forming a single structured hypothesis to answer a financial
research question, based on evidence gathered so far.

Original question: {query}

Evidence found ({evidence_count} items):
{evidence_block}

NOTE ON EVIDENCE MARKED [GRAPH PROPAGATION]: this evidence was computed
by tracing documented dependency/supply/exposure relationships across
MULTIPLE independently-extracted filings — it often reveals connections
that no single filing states outright (e.g. Company A's disclosed input
dependency + Company B's disclosed sourcing-country fact, chained
together). This is structural evidence, not a summary of one document,
and should be weighed and cited alongside filing evidence where it
supports the question — do not default to only citing plain filing text
just because it reads as more directly quotable. If propagation evidence
corroborates, extends, or is the ONLY evidence identifying a company's
exposure, say so explicitly and cite it.

Gaps — information that could NOT be found:
{gaps_block}

Contradictions found in the evidence:
{contradictions_block}

Form ONE hypothesis that best answers the question given what's actually
available. Be explicit about what limits your confidence — gaps and
contradictions should lower it, not be glossed over. Do not claim
certainty the evidence doesn't support.

Return ONLY valid JSON, no preamble, no markdown fences:

{{
  "statement": "the hypothesis, 2-4 sentences",
  "supporting_evidence_ids": ["ev1", "ev2"],
  "confidence": 0.0 to 1.0,
  "limiting_factors": ["short phrase per factor limiting confidence"]
}}
"""


def _format_evidence_block(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    lines = []
    for e in evidence:
        tag = "[GRAPH PROPAGATION] " if e.source.result_type == "propagation" else ""
        lines.append(
            f"  [{e.id}] {tag}({e.primary_entity}, {e.recency}, "
            f"reliability={e.reliability:.2f}, relevance={e.relevance:.2f}): {e.text}"
        )
    return "\n".join(lines)


def _format_gaps_block(gaps: list[Gap]) -> str:
    if not gaps:
        return "(none)"
    return "\n".join(f"  - {g.description}" for g in gaps)


def _format_contradictions_block(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "(none)"
    return "\n".join(
        f"  - {c.description}"
        + (f" (resolved: {c.resolution})" if c.resolution else " (unresolved)")
        for c in contradictions
    )


def _fallback_hypothesis(evidence: list[Evidence], gaps: list[Gap]) -> Hypothesis:
    """
    Used when there's no evidence to reason over, or the LLM call fails/
    returns something unparseable. Degrades to an honest "insufficient
    evidence" hypothesis rather than fabricating a confident-sounding one.
    """
    if not evidence:
        return Hypothesis(
            statement="No evidence was found to form a hypothesis for this question.",
            supporting_evidence_ids=[],
            confidence=0.0,
            limiting_factors=[g.description for g in gaps] or ["no evidence retrieved"],
        )
    return Hypothesis(
        statement=(
            "A hypothesis could not be formed due to a processing error — "
            "evidence exists but was not reasoned over."
        ),
        supporting_evidence_ids=[e.id for e in evidence],
        confidence=0.0,
        limiting_factors=["hypothesis formation failed — see logs"],
    )


def form_hypothesis(
    query: str,
    evidence: list[Evidence],
    gaps: list[Gap],
    contradictions: list[Contradiction],
    llm_call: LlmCallFn,
) -> Hypothesis:
    """
    Produce one Hypothesis from evidence + gaps + contradictions gathered
    so far for the original query.
    """
    if not evidence:
        log.info("No evidence available — skipping LLM call, returning empty-evidence hypothesis")
        return _fallback_hypothesis(evidence, gaps)

    prompt = HYPOTHESIS_PROMPT_TEMPLATE.format(
        query=query,
        evidence_count=len(evidence),
        evidence_block=_format_evidence_block(evidence),
        gaps_block=_format_gaps_block(gaps),
        contradictions_block=_format_contradictions_block(contradictions),
    )

    try:
        raw = llm_call(prompt)
    except Exception as e:
        log.error(f"llm_call raised during hypothesis formation: {e}")
        return _fallback_hypothesis(evidence, gaps)

    if not raw or not isinstance(raw, dict) or "statement" not in raw:
        log.warning("Hypothesis formation returned unparseable result — using fallback")
        return _fallback_hypothesis(evidence, gaps)

    # Fold in gap/contradiction descriptions directly rather than trusting
    # the model to have listed all of them itself in limiting_factors —
    # keeps the hypothesis honest even if the model under-reports.
    known_limits = [g.description for g in gaps]
    known_limits += [c.description for c in contradictions if not c.resolution]
    model_limits = raw.get("limiting_factors") or []
    if not isinstance(model_limits, list):
        model_limits = []
    limiting_factors = list(dict.fromkeys(known_limits + model_limits))  # dedupe, preserve order

    confidence = raw.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    supporting_ids = raw.get("supporting_evidence_ids") or []
    if not isinstance(supporting_ids, list):
        supporting_ids = []
    # only keep IDs that actually exist in the evidence we passed in —
    # don't let the model cite evidence that doesn't exist
    valid_ids = {e.id for e in evidence}
    supporting_ids = [i for i in supporting_ids if i in valid_ids]

    return Hypothesis(
        statement=raw["statement"],
        supporting_evidence_ids=supporting_ids,
        confidence=confidence,
        limiting_factors=limiting_factors,
    )


if __name__ == "__main__":
    from src.agents.models.contract import SourceRef

    ev1 = Evidence(
        id="ev1", sub_question_id="sq1",
        source=SourceRef(type="vector", chunk_id="c1", filing_id="f1"),
        text="AMZN 10-K discusses logistics equipment and fulfillment center capex.",
        primary_entity="AMZN", reliability=1.0, recency="2022-01-01",
        relevance=0.8, confidence=0.75,
    )
    ev2 = Evidence(
        id="ev2", sub_question_id="sq2",
        source=SourceRef(type="graph", result_type="macro_impact", chunks=["c9"]),
        text="Steel prices up 12% this month.",
        primary_entity="AMZN", reliability=0.9, recency="current",
        relevance=0.9, confidence=0.8,
    )
    gaps = [Gap(sub_question_id="sq3", description="No hedging data found for AMZN steel exposure")]

    def fake_llm_call(prompt: str) -> Optional[dict]:
        return {
            "statement": (
                "AMZN has indirect steel exposure via logistics/fulfillment capex, "
                "not primary COGS. Current steel prices are up 12%, suggesting "
                "modest margin pressure over the next 12-18 months."
            ),
            "supporting_evidence_ids": ["ev1", "ev2", "ev_fake_nonexistent"],
            "confidence": 0.75,
            "limiting_factors": ["indirect exposure inferred, not explicitly stated"],
        }

    hyp = form_hypothesis(
        "Should I be worried about AMZN's steel exposure given current commodity prices?",
        [ev1, ev2], gaps, [], fake_llm_call,
    )
    print(hyp.model_dump())

    print("\n--- empty evidence case ---")
    print(form_hypothesis("some query", [], [], [], fake_llm_call).model_dump())

    print("\n--- fallback on broken llm_call ---")
    print(form_hypothesis("some query", [ev1], [], [], lambda p: None).model_dump())