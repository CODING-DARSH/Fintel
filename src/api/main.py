import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Fintel Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    max_reflection_loops: int = 2
    k_per_hop: int = 5


class QuerySubmitResponse(BaseModel):
    query_id: str
    status: str


def _run_query_job(query_id: str, query: str, max_reflection_loops: int, k_per_hop: int):
    """
    Runs in the background (FastAPI's BackgroundTasks — same process,
    after the HTTP response has already gone out). Not a real task
    queue: fine for single-instance/low-concurrency use, but a query
    submitted here is lost if the process restarts mid-run. Worth
    upgrading to a real queue (e.g. Celery/RQ) if this needs to survive
    restarts or run many queries concurrently.
    """
    from db.agent_store import mark_query_running, complete_query, mark_query_error

    try:
        mark_query_running(query_id)

        from src.pipeline.extractor import call_llm
        from src.agents.orchestrator import run
        from src.agents.models.contract import OrchestratorRequest

        request = OrchestratorRequest(
            query=query,
            max_reflection_loops=max_reflection_loops,
            k_per_hop=k_per_hop,
        )
        log.info(f"[{query_id}] Running orchestrator for: {query!r}")
        response = run(request, call_llm)

        ok = complete_query(query_id, response)
        if ok:
            log.info(f"[{query_id}] complete")
        else:
            log.error(f"[{query_id}] orchestrator finished but complete_query() failed to persist")

    except Exception as e:
        log.exception(f"[{query_id}] job failed")
        mark_query_error(query_id, str(e))


@app.post("/query", response_model=QuerySubmitResponse)
def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    from db.agent_store import create_pending_query

    query_id = create_pending_query(req.query, req.max_reflection_loops, req.k_per_hop)
    if query_id is None:
        raise HTTPException(status_code=503, detail="Database not reachable — could not create query")

    background_tasks.add_task(
        _run_query_job, query_id, req.query, req.max_reflection_loops, req.k_per_hop
    )

    return QuerySubmitResponse(query_id=query_id, status="pending")


@app.get("/queries/{query_id}")
def get_query(query_id: str):
    from db.agent_store import get_response

    result = get_response(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return result


@app.get("/queries")
def list_queries_endpoint(limit: int = 50, offset: int = 0):
    from db.agent_store import list_queries

    if limit > 200:
        limit = 200
    return {"queries": list_queries(limit=limit, offset=offset)}


@app.get("/health")
def health():
    from src.db.database import db_available

    return {"postgres": db_available()}