from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from src.agents.models.contract import SubQuestion

log = logging.getLogger(__name__)

LlmCallFn = Callable[[str], Optional[dict]]

# retrieval_focus is intentionally open-ended (a short free-text hint, not
# an enum) — it's passed into gather_evidence as a bias/label, not routed
# to a fixed set of services, so new focuses (e.g. "regulatory", "labor",
# "currency") don't require a code change here. Only bare structural
# validity is checked (non-empty string), not membership in a fixed list.
DEFAULT_FOCUS = "filing"


DECOMPOSE_PROMPT_TEMPLATE = """You are decomposing a financial research question into
smaller sub-questions that can each be answered by retrieval.

Question: {query}

Break this into 2-5 sub-questions that together would let someone answer
the original question with evidence. For each sub-question, assign a
retrieval_focus — a short lowercase label describing what kind of evidence
it needs (e.g. filing, macro, news, market, competitor, regulatory,
currency, labor, or any other label that fits). Don't force-fit a category
if none of the common ones apply — invent a short accurate one instead.

Return ONLY valid JSON, no preamble, no markdown fences, matching this shape:

{{
  "sub_questions": [
    {{"text": "...", "retrieval_focus": "filing"}},
    {{"text": "...", "retrieval_focus": "macro"}}
  ]
}}
"""


def build_decompose_prompt(query: str) -> str:
    return DECOMPOSE_PROMPT_TEMPLATE.format(query=query)


def _fallback_sub_questions(query: str) -> list[SubQuestion]:
    """
    If the model call fails or returns something unparseable, don't crash
    the pipeline — fall back to treating the original query as a single
    sub-question with a generic focus. Multi-hop reasoning degrades to
    single-hop rather than failing outright.
    """
    log.warning("Decomposition failed or unparseable — falling back to single sub-question")
    return [SubQuestion(id="sq1", text=query, retrieval_focus=DEFAULT_FOCUS)]


def decompose_query(query: str, llm_call: LlmCallFn) -> list[SubQuestion]:
    """
    Turn a raw query into a list of SubQuestion objects.

    Args:
        query    : the original user question
        llm_call : any function matching call_llm(prompt: str) -> Optional[dict],
                   e.g. extractor.call_llm from your existing Gemini/Groq rotation

    Returns:
        list[SubQuestion] — always at least one (falls back to the original
        query as a single sub-question if the LLM call fails or returns
        something that doesn't parse).
    """
    prompt = build_decompose_prompt(query)

    try:
        raw = llm_call(prompt)
    except Exception as e:
        log.error(f"llm_call raised during decomposition: {e}")
        return _fallback_sub_questions(query)

    if not raw:
        return _fallback_sub_questions(query)

    items = raw.get("sub_questions") if isinstance(raw, dict) else None
    if not items or not isinstance(items, list):
        return _fallback_sub_questions(query)

    sub_questions = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "text" not in item:
            continue
        focus = item.get("retrieval_focus") or DEFAULT_FOCUS
        if not isinstance(focus, str) or not focus.strip():
            focus = DEFAULT_FOCUS
        focus = focus.strip().lower()
        sub_questions.append(
            SubQuestion(id=f"sq{i+1}", text=item["text"], retrieval_focus=focus)
        )

    if not sub_questions:
        return _fallback_sub_questions(query)

    return sub_questions


if __name__ == "__main__":
    # Manual smoke test with a fake llm_call — no real API hit.
    # Run: python src/agents/decompose.py
    def fake_llm_call(prompt: str) -> Optional[dict]:
        return {
            "sub_questions": [
                {"text": "Does AMZN have steel exposure at all?", "retrieval_focus": "filing"},
                {"text": "What is the current steel price and trend?", "retrieval_focus": "macro"},
                {"text": "Has AMZN flagged steel as a material risk?", "retrieval_focus": "filing"},
            ]
        }

    result = decompose_query(
        "Should I be worried about AMZN's steel exposure given current commodity prices?",
        fake_llm_call,
    )
    for sq in result:
        print(sq.model_dump())

    print("\n--- fallback test (simulating a broken llm_call) ---")
    result2 = decompose_query("Some query", lambda p: None)
    for sq in result2:
        print(sq.model_dump())