# =============================================================================
# run_agent_query.py
# =============================================================================
# The actual "how do I run this for real" entry point.
#
# Wires:
#   - src.pipeline.extractor.call_llm   -> your real Gemini+Groq 6-key
#                                          rotation (already built, already
#                                          used by the extraction pipeline)
#   - GraphRetriever / VectorRetriever  -> real Neo4j / ChromaDB, using the
#                                          SAME env vars docker-compose.yml
#                                          already sets on the `pipeline`
#                                          service (NEO4J_HOST, CHROMA_HOST,
#                                          etc.) — nothing new to configure
#   - src.agents.orchestrator.run       -> the full agent stack built so far
#
# NOT changed here: docker-compose.yml, config.py, graph_retriever.py's env
# var handling, extractor.py's key rotation — all of that was already
# correct and didn't need touching. This file is the one missing piece:
# something that actually calls everything together in one process.
#
# Usage (from inside the pipeline container, or anywhere with the same
# env vars set — NEO4J_HOST, CHROMA_HOST, GEMINI_API_KEY_1.., GROQ_API_KEY_1..):
#
#   docker compose run --rm pipeline python run_agent_query.py \
#       "Should I be worried about AMZN's steel exposure given current commodity prices?"
#
# or, without Docker, as long as Neo4j/Chroma are reachable at the hosts
# your .env points to:
#
#   python run_agent_query.py "your question here"
# =============================================================================

import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_agent_query")


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_agent_query.py "your question here" [max_reflection_loops] [k_per_hop]')
        sys.exit(1)

    query = sys.argv[1]
    max_reflection_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    k_per_hop = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    # --- Real LLM call: your existing Gemini+Groq rotation -----------------
    # This is the exact function used by src/pipeline/extractor.py already —
    # same key rotation, same call_llm(prompt) -> Optional[dict] signature
    # every agent step was built against. No wrapper needed.
    try:
        from src.pipeline.extractor import call_llm
    except Exception as e:
        log.error(
            f"Could not import call_llm from src.pipeline.extractor: {e}\n"
            f"Make sure GEMINI_API_KEY_1/GROQ_API_KEY_1 etc. are set in your "
            f".env — extractor.py raises if no keys are found at all."
        )
        sys.exit(1)

    # --- Real retrievers: same GraphRetriever/VectorRetriever used by ------
    # test_retrieval.py and gather_evidence.py — connection details come
    # from env vars already set by docker-compose.yml on the `pipeline`
    # service (NEO4J_HOST=neo4j, CHROMA_HOST=chromadb, etc.), so nothing
    # extra to configure when run via `docker compose run pipeline ...`.
    from src.retrieval import GraphRetriever, VectorRetriever, RetrievalMerger, QueryParser
    from src.agents.orchestrator import run
    from src.agents.models.contract import OrchestratorRequest

    graph  = GraphRetriever()
    vector = VectorRetriever()

    # quick reachability check before running the full loop — fail fast
    # with a clear message rather than silently degrading through every
    # fallback path in the agent stack and producing a confusing
    # "no evidence found" answer with no explanation why.
    try:
        graph._get_driver().verify_connectivity()
        log.info("Neo4j: connected")
    except Exception as e:
        log.warning(f"Neo4j NOT reachable ({e}) — graph evidence will be empty, not an error in the agent logic itself.")

    try:
        vector.retrieve(query="connectivity check", top_k=1)
        log.info("ChromaDB: reachable")
    except Exception as e:
        log.warning(f"ChromaDB NOT reachable ({e}) — vector evidence will be empty, not an error in the agent logic itself.")

    request = OrchestratorRequest(
        query=query,
        max_reflection_loops=max_reflection_loops,
        k_per_hop=k_per_hop,
    )

    log.info(f"Running orchestrator for: {query!r}")
    response = run(request, call_llm)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.answer)
    print()
    print("REASONING CHAIN:", response.reasoning_chain)
    print("CONFIDENCE:", response.confidence)
    print("STOPPED REASON:", response.stopped_reason, "| REFLECTIONS:", response.reflection_count)
    print()
    print("SUB-QUESTIONS:")
    for sq in response.sub_questions:
        print(f"  [{sq.retrieval_focus}] {sq.text}")
    print()
    if response.gaps:
        print("GAPS:")
        for g in response.gaps:
            print(f"  - {g.description}")
        print()
    if response.contradictions:
        print("CONTRADICTIONS:")
        for c in response.contradictions:
            status = f"resolved: {c.resolution}" if c.resolution else "unresolved"
            print(f"  - {c.description} ({status})")
        print()
    print("FOLLOW-UP QUESTIONS:")
    for q in response.follow_up_questions:
        print(f"  - {q}")

    # Full structured dump too, in case you want to pipe this into a file
    # or inspect the raw Evidence/citations for debugging.
    out_path = "data/eval/last_agent_response.json"
    try:
        with open(out_path, "w") as f:
            json.dump(response.model_dump(), f, indent=2)
        log.info(f"Full structured response written to {out_path}")
    except Exception as e:
        log.warning(f"Could not write structured response to {out_path}: {e}")

    graph.close()


if __name__ == "__main__":
    main()
    
    
    