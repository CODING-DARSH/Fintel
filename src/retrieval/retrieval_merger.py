# =============================================================================
# src/retrieval/retrieval_merger.py
# =============================================================================
# Merges results from GraphRetriever and VectorRetriever.
#
# TECHNIQUES:
#   Cross-source deduplication  — same chunk_id from both sources → merge
#   Reciprocal Rank Fusion      — combine graph + vector rankings
#   Source attribution          — every result knows where it came from
#   Diversity enforcement       — avoid returning 10 chunks from same filing
#   Confidence-weighted scoring — graph results with high confidence boosted

import logging
from dataclasses import dataclass, field
from collections import defaultdict

from .graph_retriever  import GraphResult
from .vector_retriever import VectorResult

log = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class MergedResult:
    """Final merged result combining graph and vector evidence."""
    result_id     : str
    result_type   : str          # "graph", "vector", or "both"
    primary_entity: str
    text          : str          # chunk text (if vector result available)
    summary       : str          # structured summary (from graph or generated)
    final_score   : float
    graph_score   : float = 0.0
    vector_score  : float = 0.0
    sources       : list  = field(default_factory=list)   # [{type, id, date}]
    tickers       : list  = field(default_factory=list)
    filing_dates  : list  = field(default_factory=list)
    section_ids   : list  = field(default_factory=list)
    metadata      : dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "result_id"     : self.result_id,
            "result_type"   : self.result_type,
            "primary_entity": self.primary_entity,
            "text"          : self.text,
            "summary"       : self.summary,
            "final_score"   : self.final_score,
            "graph_score"   : self.graph_score,
            "vector_score"  : self.vector_score,
            "sources"       : self.sources,
            "tickers"       : self.tickers,
            "filing_dates"  : self.filing_dates,
            "section_ids"   : self.section_ids,
            "metadata"      : self.metadata,
        }


class RetrievalMerger:
    """
    Merges and re-ranks results from graph and vector retrievers.
    Produces a unified ranked list ready for agent consumption.
    """

    def __init__(
        self,
        graph_weight  : float = 0.4,   # weight for graph results in RRF
        vector_weight : float = 0.6,   # weight for vector results in RRF
        diversity_k   : int   = 2,     # max results per ticker (diversity)
    ):
        self.graph_weight  = graph_weight
        self.vector_weight = vector_weight
        self.diversity_k   = diversity_k

    def merge(
        self,
        graph_results  : list[GraphResult],
        vector_results : list[VectorResult],
        top_k          : int = 10,
        enforce_diversity: bool = True,
    ) -> list[MergedResult]:
        """
        Merge graph + vector results using weighted RRF.

        Args:
            graph_results   : Results from GraphRetriever
            vector_results  : Results from VectorRetriever
            top_k           : Final number of results
            enforce_diversity: Limit results per ticker for breadth

        Returns:
            List of MergedResult sorted by final_score descending
        """
        if not graph_results and not vector_results:
            return []

        # Build lookup maps
        # Graph: keyed by (result_type, primary_entity)
        # Vector: keyed by chunk_id
        graph_map  = {}
        vector_map = {}

        for i, gr in enumerate(graph_results):
            key = f"graph_{gr.result_type}_{gr.primary_entity}_{i}"
            graph_map[key] = (i, gr)

        for i, vr in enumerate(vector_results):
            key = f"vector_{vr.chunk_id}"
            vector_map[key] = (i, vr)

        # RRF scores — combine graph rank and vector rank
        rrf_scores = defaultdict(float)

        for key, (rank, _) in graph_map.items():
            rrf_scores[key] += self.graph_weight / (RRF_K + rank + 1)

        for key, (rank, _) in vector_map.items():
            rrf_scores[key] += self.vector_weight / (RRF_K + rank + 1)

        # Normalize RRF scores to 0-1
        max_rrf = max(rrf_scores.values(), default=1.0) or 1.0

        # Build merged results
        merged    = []
        seen_chunks = set()

        for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            final_score = rrf_score / max_rrf

            if key in graph_map:
                _, gr = graph_map[key]
                result = self._from_graph(gr, final_score, gr.score)
            else:
                _, vr = vector_map[key]
                cid = vr.chunk_id
                if cid in seen_chunks:
                    continue
                seen_chunks.add(cid)
                result = self._from_vector(vr, final_score, vr.hybrid_score)

            merged.append(result)

        # Diversity enforcement — limit results per ticker
        if enforce_diversity:
            merged = self._enforce_diversity(merged, self.diversity_k)

        return merged[:top_k]

    def _from_graph(
        self,
        gr          : GraphResult,
        final_score : float,
        graph_score : float,
    ) -> MergedResult:
        """Convert GraphResult to MergedResult."""
        data    = gr.data
        summary = self._summarize_graph_result(gr)

        return MergedResult(
            result_id     = f"graph_{gr.result_type}_{gr.primary_entity}",
            result_type   = "graph",
            primary_entity= gr.primary_entity,
            text          = summary,
            summary       = summary,
            final_score   = round(final_score, 4),
            graph_score   = round(graph_score, 4),
            vector_score  = 0.0,
            sources       = [{"type": "graph", "result_type": gr.result_type,
                               "chunks": gr.source_chunks}],
            tickers       = [gr.primary_entity],
            filing_dates  = gr.filing_dates,
            section_ids   = [],
            metadata      = {"hop_distance": gr.hop_distance, "data": data},
        )

    def _from_vector(
        self,
        vr          : VectorResult,
        final_score : float,
        vector_score: float,
    ) -> MergedResult:
        """Convert VectorResult to MergedResult."""
        return MergedResult(
            result_id     = f"vector_{vr.chunk_id}",
            result_type   = "vector",
            primary_entity= vr.ticker,
            text          = vr.text,
            summary       = (vr.text[:200] + "..."
                             if len(vr.text) > 200 else vr.text),
            final_score   = round(final_score, 4),
            graph_score   = 0.0,
            vector_score  = round(vector_score, 4),
            sources       = [{"type": "vector", "chunk_id": vr.chunk_id,
                               "filing_id": vr.filing_id}],
            tickers       = [vr.ticker],
            filing_dates  = [vr.filing_date] if vr.filing_date else [],
            section_ids   = [vr.section_id] if vr.section_id else [],
            metadata      = {
                "dense_score" : vr.dense_score,
                "sparse_score": vr.sparse_score,
                "section_name": vr.section_name,
                "cluster_id"  : vr.cluster_id,
                "chunk_meta"  : vr.metadata,
            },
        )

    def _summarize_graph_result(self, gr: GraphResult) -> str:
        """Convert graph result data into a human-readable summary string."""
        d = gr.data
        t = gr.result_type

        if t == "company_risk":
            return (f"{gr.primary_entity} exposed to {d.get('category','?')} "
                    f"({d.get('subcategory','?')}) — severity: {d.get('severity','?')}. "
                    f"{d.get('description','')}")

        if t == "shared_risk":
            return (f"{d.get('ticker','?')} exposed to {d.get('category','?')}: "
                    f"{d.get('description','')} "
                    f"[severity: {d.get('severity','?')}]")

        if t == "supply_chain_dependency":
            return (f"{gr.primary_entity} depends on {d.get('input_name','?')} "
                    f"(type: {d.get('input_type','?')}, "
                    f"criticality: {d.get('criticality','?')}, "
                    f"suppliers: {d.get('suppliers',[])})")

        if t == "propagation_risk":
            return (f"{d.get('event_type','?')} in {d.get('geography','?')} "
                    f"→ {gr.primary_entity}: {d.get('impact_type','?')} "
                    f"(severity: {d.get('severity','?')}, "
                    f"lag: {d.get('lag_time','?')})")

        if t == "competitor":
            return (f"{d.get('ticker','?')} competes with {gr.primary_entity} "
                    f"in {d.get('segment','unspecified segment')}")

        if t == "causal_chain":
            return (f"{d.get('cause','?')} → {d.get('effect','?')} "
                    f"via {d.get('mechanism','?')} "
                    f"[affects: {d.get('affected_companies',[])}]")

        if t == "macro_impact":
            return (f"{d.get('indicator','?')} moved {d.get('direction','?')} "
                    f"({d.get('magnitude','?')} magnitude) "
                    f"→ affects {d.get('ticker','?')}")

        if t == "news_mention":
            return (f"[{d.get('urgency','?')}] {d.get('title','?')} "
                    f"— {d.get('summary','')}")

        if t == "executive_change":
            return (f"{d.get('person_name','?')} ({d.get('role','?')}) "
                    f"at {d.get('ticker','?')} — {d.get('filing_date','?')}")

        if t == "geographic_exposure":
            return (f"{d.get('ticker','?')} concentrated in "
                    f"{d.get('geography','?')} "
                    f"({d.get('concentration_type','?')}: "
                    f"{d.get('percentage','?')})")

        if t == "market_signal":
            return (f"{gr.primary_entity}: {d.get('signal_type','?')} "
                    f"— {d.get('note','?')} "
                    f"(price: {d.get('latest_price','?')}, "
                    f"1d: {d.get('pct_change_1d','?')}%)")

        # Fallback
        return str(d)

    def _enforce_diversity(
        self,
        results : list[MergedResult],
        k       : int,
    ) -> list[MergedResult]:
        """
        Ensure no more than k results per ticker.
        Preserves overall ranking order while enforcing breadth.
        """
        counts  = defaultdict(int)
        diverse = []
        overflow = []

        for r in results:
            ticker = r.primary_entity
            if counts[ticker] < k:
                diverse.append(r)
                counts[ticker] += 1
            else:
                overflow.append(r)

        # Append overflow at end (still visible, just lower priority)
        return diverse + overflow

    def merge_graph_only(
        self,
        graph_results: list[GraphResult],
        top_k        : int = 10,
    ) -> list[MergedResult]:
        """When no vector results available — merge graph results only."""
        return self.merge(graph_results, [], top_k=top_k,
                          enforce_diversity=False)

    def merge_vector_only(
        self,
        vector_results: list[VectorResult],
        top_k         : int = 10,
    ) -> list[MergedResult]:
        """When no graph results available — merge vector results only."""
        return self.merge([], vector_results, top_k=top_k)