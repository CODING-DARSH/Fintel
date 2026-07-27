# =============================================================================
# src/agents/models/contract.py
# =============================================================================
# STEP 1 of the agent build: the data contract only.
#
# No LLM calls, no orchestration logic here — just the shapes that every
# later piece (decomposition, evidence gathering, synthesis) will read from
# and write into. Get this right first; everything else just fills it in.
#
# Maps directly onto the design agreed in conversation:
#   SubQuestion   -> decomposition step
#   Evidence      -> parallel research / evidence evaluation steps
#   Contradiction -> contradiction detection step
#   Gap           -> gap detection step
#   Hypothesis    -> hypothesis formation step
#   OrchestratorResponse -> synthesis step (final object returned)
#
# Evidence.source and Evidence.text are pass-throughs of what already exists
# in retrieval_merger.MergedResult (.sources, .text) — nothing new is
# invented here, this just gives the agent layer its own typed view of it.
# =============================================================================

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class OrchestratorRequest(BaseModel):
    query: str
    max_reflection_loops: int = 2
    k_per_hop: int = 5


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

class SubQuestion(BaseModel):
    id: str
    text: str
    # A retrieval hint, NOT a separate service/agent to call — retrieval
    # still goes through the one retrieval_merger.merge() call per
    # sub-question. This just biases/filters that call
    # (e.g. "macro", "filing", "news", "competitor", "market").
    retrieval_focus: str


# ---------------------------------------------------------------------------
# Evidence  (one MergedResult -> one Evidence, evaluated)
# ---------------------------------------------------------------------------

class SourceRef(BaseModel):
    """Pass-through of MergedResult.sources[i] — filing chunk or graph node."""
    type: Literal["graph", "vector"]
    # vector fields
    chunk_id: Optional[str] = None
    filing_id: Optional[str] = None
    # graph fields
    result_type: Optional[str] = None
    chunks: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    sub_question_id: str
    source: SourceRef
    text: str                       # full chunk / summary text — NOT truncated,
                                     # future UI will need to show it verbatim
    primary_entity: str              # ticker or entity name
    filing_dates: list[str] = Field(default_factory=list)

    # evidence evaluation fields
    reliability: float               # from config.SOURCE_TRUST, by source type
    recency: str                     # a date string, or "current" for live signals
    relevance: float                 # does this evidence actually answer
                                      # the sub-question (0-1)
    confidence: float                # combined score for this piece of evidence


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

class Contradiction(BaseModel):
    evidence_ids: list[str]
    description: str
    resolution: Optional[str] = None   # how/whether it was reconciled


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

class Gap(BaseModel):
    sub_question_id: str
    description: str


# ---------------------------------------------------------------------------
# Hypothesis formation
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    statement: str
    supporting_evidence_ids: list[str]
    confidence: float
    limiting_factors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph propagation analytics (src/agents/analytics/graph_propagation.py)
#
# Distinct from Evidence/SourceRef above — these describe the STRUCTURE of
# a propagation search before it's converted into an Evidence object.
# PropagationResult.target_found is deliberately explicit (not inferred
# from an empty list) — "no documented connection found" must be exactly
# as visible a result as "connection found, confidence 0.56".
# ---------------------------------------------------------------------------

class PathStep(BaseModel):
    from_node: str
    from_type: str          # "Input" | "Geography" | "Event" | "Company"
    relation: str             # e.g. "SOURCED_FROM", "DEPENDS_ON", "SUPPLIES_TO"
    to_node: str
    to_type: str
    edge_confidence: Optional[float] = None   # real confidence property, if present
    edge_weight: float = 0.0                    # resolved numeric weight (see graph_propagation.py)
    weight_source: str = "unknown"               # "numeric_field" | "categorical_map" | "relation_default"


class PropagationPath(BaseModel):
    steps: list[PathStep]
    target_ticker: str
    path_score: float
    depth: int


class PropagationResult(BaseModel):
    start_entity: str
    start_entity_type: str
    companies_reached: list[str] = Field(default_factory=list)
    paths: dict[str, list[PropagationPath]] = Field(default_factory=dict)  # ticker -> paths
    target_ticker: Optional[str] = None
    target_found: bool = False


# ---------------------------------------------------------------------------
# Final response (synthesis)
# ---------------------------------------------------------------------------

class OrchestratorResponse(BaseModel):
    query: str

    sub_questions: list[SubQuestion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    hypothesis: Optional[Hypothesis] = None

    answer: str = ""
    reasoning_chain: str = ""
    citations: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0
    follow_up_questions: list[str] = Field(default_factory=list)

    reflection_count: int = 0
    stopped_reason: Literal[
        "synthesized",
        "max_reflections",
        "no_new_evidence",
        "not_run",
    ] = "not_run"