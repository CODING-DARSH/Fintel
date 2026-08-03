# =============================================================================
# src/agents/orchestrator.py
# =============================================================================
# STEP 10 — final glue. Every piece this calls (steps 2-9) is already
# independently tested; this file just wires them into the public
# entry point:
#
#       run(request: OrchestratorRequest, llm_call) -> OrchestratorResponse
#
# Flow:
#   OrchestratorRequest
#     -> run_reasoning_loop   (decompose -> [gather -> gaps -> contradictions
#                               -> hypothesis -> reflect]* -> settled state)
#     -> synthesize           (settled state -> answer/reasoning_chain/
#                               citations/follow_up_questions)
#     -> OrchestratorResponse (everything assembled)
#
# No new logic lives here beyond assembling the final response object —
# if something's wrong with decomposition, retrieval, contradictions,
# hypothesis, reflection, or synthesis, the bug is in that step's own
# file, not this one.
# =============================================================================

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.agents.models.contract import OrchestratorRequest, OrchestratorResponse
from src.agents.run_reasoning_loop import run_reasoning_loop
from src.agents.synthesize import synthesize

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]


def run(request: OrchestratorRequest, llm_call: LlmCallFn) -> OrchestratorResponse:
    """
    The single public entry point for the agent stack.

    Args:
        request  : OrchestratorRequest — the query + loop/retrieval limits
        llm_call : call_llm(prompt: str) -> Optional[dict] — your existing
                   Groq/Gemini rotation function, passed straight through
                   to every step that needs it (decompose, contradictions,
                   hypothesis, reflect, synthesis).

    Returns:
        OrchestratorResponse — fully populated: sub_questions, evidence,
        contradictions, gaps, hypothesis, answer, reasoning_chain,
        citations, confidence, follow_up_questions, reflection_count,
        stopped_reason.
    """
    log.info(f"Orchestrator run starting: {request.query!r}")

    loop_result = run_reasoning_loop(
        query=request.query,
        llm_call=llm_call,
        max_reflection_loops=request.max_reflection_loops,
        k_per_hop=request.k_per_hop,
    )

    synthesis_result = synthesize(
        query=request.query,
        evidence=loop_result.evidence,
        gaps=loop_result.gaps,
        contradictions=loop_result.contradictions,
        hypothesis=loop_result.hypothesis,
        llm_call=llm_call,
    )

    response = OrchestratorResponse(
        query=request.query,
        sub_questions=loop_result.sub_questions,
        evidence=loop_result.evidence,
        contradictions=loop_result.contradictions,
        gaps=loop_result.gaps,
        hypothesis=loop_result.hypothesis,
        answer=synthesis_result.answer,
        reasoning_chain=synthesis_result.reasoning_chain,
        citations=synthesis_result.citations,
        confidence=loop_result.hypothesis.confidence,
        follow_up_questions=synthesis_result.follow_up_questions,
        reflection_count=loop_result.reflection_count,
        stopped_reason=loop_result.stopped_reason,
    )

    log.info(
        f"Orchestrator run complete: stopped_reason={response.stopped_reason}, "
        f"reflection_count={response.reflection_count}, "
        f"confidence={response.confidence:.2f}, "
        f"evidence={len(response.evidence)}, gaps={len(response.gaps)}, "
        f"contradictions={len(response.contradictions)}"
    )

    return response


if __name__ == "__main__":
    # Full end-to-end structural smoke test — fake llm_call plays every
    # role. No live Neo4j/Chroma needed; gather_all_evidence will come
    # back empty in this sandbox, which exercises the empty-evidence
    # fallback paths all the way through to a real OrchestratorResponse.

    hop_count = {"n": 0}

    def fake_llm_call(prompt: str) -> Optional[dict]:
        if "decomposing a financial research question" in prompt:
            return {"sub_questions": [
                {"text": "Does AMZN have steel exposure?", "retrieval_focus": "filing"},
                {"text": "What is the current steel price trend?", "retrieval_focus": "macro"},
            ]}
        if "checking two pieces of financial evidence" in prompt:
            return {"is_contradiction": False, "description": "", "resolution": ""}
        if "forming a single structured hypothesis" in prompt:
            return {
                "statement": "No strong evidence available in this environment to form a confident view.",
                "supporting_evidence_ids": [],
                "confidence": 0.2,
                "limiting_factors": ["no evidence retrieved — sandbox has no live retrieval backends"],
            }
        if "deciding whether a financial research answer is ready" in prompt:
            hop_count["n"] += 1
            if hop_count["n"] >= 2:
                return {"needs_another_round": False, "reason": "reached smoke-test cap"}
            return {
                "needs_another_round": True,
                "reason": "confidence too low",
                "new_sub_question": f"Follow-up question round {hop_count['n']}",
                "new_sub_question_focus": "macro",
            }
        if "writing the final answer" in prompt:
            return {
                "answer": (
                    "There isn't enough evidence in this environment to give a confident "
                    "answer — no retrieval backend is reachable, so this reflects the "
                    "system's fallback behavior rather than a real financial conclusion."
                ),
                "reasoning_chain": "no evidence retrieved -> no hypothesis support -> low-confidence answer",
                "follow_up_questions": ["Retry once Neo4j/Chroma are reachable."],
            }
        return None

    from src.agents.models.contract import OrchestratorRequest

    req = OrchestratorRequest(
        query="Should I be worried about AMZN's steel exposure given current commodity prices?",
        max_reflection_loops=2,
        k_per_hop=3,
    )
    resp = run(req, fake_llm_call)

    print("=" * 70)
    print("QUERY:", resp.query)
    print("=" * 70)
    print("ANSWER:", resp.answer)
    print()
    print("REASONING CHAIN:", resp.reasoning_chain)
    print("CONFIDENCE:", resp.confidence)
    print("STOPPED REASON:", resp.stopped_reason)
    print("REFLECTION COUNT:", resp.reflection_count)
    print()
    print("SUB-QUESTIONS:", [sq.text for sq in resp.sub_questions])
    print("GAPS:", [g.description for g in resp.gaps])
    print("CONTRADICTIONS:", len(resp.contradictions))
    print("CITATIONS:", len(resp.citations))
    print("FOLLOW-UPS:", resp.follow_up_questions)
    print()
    print("Full response validates as OrchestratorResponse:", type(resp).__name__)