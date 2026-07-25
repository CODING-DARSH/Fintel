# =============================================================================
# src/retrieval/vector_retriever.py
# =============================================================================
# Retrieves relevant chunks from ChromaDB using hybrid retrieval:
#
# TECHNIQUES:
#   Two-level retrieval:
#     Level 1 — search cluster_heads (~100 vectors, fast coarse filter)
#     Level 2 — search member chunks within matched clusters (precise)
#   Dense retrieval  — cosine similarity on BAAI/bge-base-en-v1.5 embeddings
#   Sparse retrieval — BM25 keyword matching on chunk text
#   Hybrid fusion    — combine dense + sparse scores via RRF
#   Metadata filtering — filter by ticker, date, section BEFORE vector search
#                        (massive speedup: search 500 chunks not 15,000)

import os
import re
import logging
import math
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

COLLECTION_CHUNKS = "chunks"
COLLECTION_HEADS  = "cluster_heads"

# RRF constant — controls how much rank position matters
# Higher k = smoother fusion, less sensitive to top-1 differences
RRF_K = 60


@dataclass
class VectorResult:
    """Single result from vector retrieval."""
    chunk_id     : str
    ticker       : str
    filing_id    : str
    filing_type  : str
    filing_date  : str
    section_id   : str
    section_name : str
    text         : str
    dense_score  : float = 0.0   # cosine similarity score
    sparse_score : float = 0.0   # BM25 score
    hybrid_score : float = 0.0   # RRF combined score
    cluster_id   : int   = -1
    metadata     : dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id"    : self.chunk_id,
            "ticker"      : self.ticker,
            "filing_id"   : self.filing_id,
            "filing_type" : self.filing_type,
            "filing_date" : self.filing_date,
            "section_id"  : self.section_id,
            "section_name": self.section_name,
            "text"        : self.text,
            "dense_score" : self.dense_score,
            "sparse_score": self.sparse_score,
            "hybrid_score": self.hybrid_score,
            "cluster_id"  : self.cluster_id,
            "metadata"    : self.metadata,
        }


class VectorRetriever:
    """
    Hybrid vector retriever combining dense embeddings and BM25 sparse search.
    Uses two-level retrieval (cluster heads → member chunks) for speed.
    """

    def __init__(self):
        self._chroma       = None
        self._chunks_col   = None
        self._heads_col    = None
        self._model        = None

    # -------------------------------------------------------------------------
    # Client initialization
    # -------------------------------------------------------------------------

    def _get_chroma(self):
        if self._chroma is None:
            import chromadb
            self._chroma = chromadb.HttpClient(
                host=CHROMA_HOST, port=CHROMA_PORT
            )
        return self._chroma

    def _get_chunks(self):
        if self._chunks_col is None:
            self._chunks_col = self._get_chroma().get_or_create_collection(
                name=COLLECTION_CHUNKS,
                metadata={"hnsw:space": "cosine"},
            )
        return self._chunks_col

    def _get_heads(self):
        if self._heads_col is None:
            try:
                self._heads_col = self._get_chroma().get_collection(
                    name=COLLECTION_HEADS
                )
            except Exception:
                self._heads_col = None
        return self._heads_col

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model

    def _embed(self, text: str) -> list:
        return self._get_model().encode([text], show_progress_bar=False)[0].tolist()

    # -------------------------------------------------------------------------
    # BM25 sparse scoring
    # -------------------------------------------------------------------------

    def _tokenize(self, text: str) -> list:
        """Simple tokenizer for BM25."""
        text   = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text)
        stops  = {
            "the","and","for","are","but","not","you","all","can","her",
            "was","one","our","out","day","get","has","him","his","how",
            "its","may","now","she","was","will","with","this","that","they",
            "have","from","been","were","said","each","which","their","time",
            "than","then","them","into","your","some","could","would","other",
        }
        return [t for t in tokens if t not in stops]

    def _bm25_score(
        self,
        query_tokens : list,
        doc_text     : str,
        avg_doc_len  : float = 200.0,
        k1           : float = 1.5,
        b            : float = 0.75,
    ) -> float:
        """
        BM25 score for a single document.
        Captures exact keyword matches that dense retrieval misses.
        """
        doc_tokens  = self._tokenize(doc_text)
        doc_len     = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        freq_map = defaultdict(int)
        for t in doc_tokens:
            freq_map[t] += 1

        score = 0.0
        # Approximate IDF: use fixed N=15000 total docs
        N = 15000
        for token in query_tokens:
            tf  = freq_map.get(token, 0)
            if tf == 0:
                continue
            # BM25 TF component
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            )
            # Approximate IDF — assume 5% of docs contain each query term
            idf = math.log((N - 0.5) / (N * 0.05 + 0.5) + 1)
            score += idf * tf_norm

        return score

    # -------------------------------------------------------------------------
    # RRF fusion
    # -------------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_ranked  : list,   # [(chunk_id, score), ...]
        sparse_ranked : list,
        k             : int = RRF_K,
    ) -> dict:
        """
        Reciprocal Rank Fusion — combines dense and sparse rankings.
        RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        Robust to score scale differences between dense and sparse.
        """
        rrf_scores = defaultdict(float)

        for rank, (cid, _) in enumerate(dense_ranked):
            rrf_scores[cid] += 1.0 / (k + rank + 1)

        for rank, (cid, _) in enumerate(sparse_ranked):
            rrf_scores[cid] += 1.0 / (k + rank + 1)

        return dict(rrf_scores)

    # -------------------------------------------------------------------------
    # Two-level retrieval
    # -------------------------------------------------------------------------

    def _get_relevant_cluster_ids(
        self,
        query_embedding : list,
        n_clusters      : int = 5,
    ) -> list:
        """
        Level 1: search cluster heads to find relevant topic clusters.
        Fast — only searches ~100 vectors instead of 15,000.
        Returns list of cluster_ids to search in Level 2.
        """
        heads = self._get_heads()
        if heads is None:
            return []  # fall back to full search if no cluster heads

        try:
            results = heads.query(
                query_embeddings=[query_embedding],
                n_results=min(n_clusters, heads.count()),
                include=["metadatas", "distances"],
            )
            cluster_ids = []
            for meta in results.get("metadatas", [[]])[0]:
                cid = meta.get("cluster_id")
                if cid is not None and cid != -1:
                    cluster_ids.append(int(cid))
            return cluster_ids
        except Exception as e:
            log.debug(f"Cluster head search failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Core retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query        : str,
        tickers      : Optional[list] = None,
        sections     : Optional[list] = None,
        filing_types : Optional[list] = None,
        date_from    : Optional[str]  = None,
        date_to      : Optional[str]  = None,
        top_k        : int = 10,
        use_clusters : bool = True,
        alpha        : float = 0.6,   # weight for dense (1-alpha for sparse)
    ) -> list[VectorResult]:
        """
        Main retrieval method. Hybrid dense + sparse with two-level cluster search.

        Args:
            query        : Natural language query
            tickers      : Filter to specific tickers
            sections     : Filter to specific section IDs (e.g. ["1A", "7"])
            filing_types : Filter to "10-K" or "8-K"
            date_from    : Filter filings from this date (YYYY-MM-DD)
            date_to      : Filter filings to this date
            top_k        : Number of results to return
            use_clusters : Whether to use two-level cluster search
            alpha        : Weight for dense score in hybrid (0=sparse, 1=dense)

        Returns:
            List of VectorResult sorted by hybrid_score descending
        """
        collection = self._get_chunks()
        total      = collection.count()
        if total == 0:
            log.warning("ChromaDB chunks collection is empty")
            return []

        # 1. Embed query
        query_embedding = self._embed(query)
        query_tokens    = self._tokenize(query)

        # 2. Build metadata filter
        where_clause = self._build_where(
            tickers, sections, filing_types, date_from, date_to
        )

        # 3. Two-level retrieval via cluster heads
        candidate_ids = []
        if use_clusters:
            cluster_ids = self._get_relevant_cluster_ids(query_embedding)
            if cluster_ids:
                # Add cluster_id filter to where clause
                cluster_where = {"cluster_id": {"$in": cluster_ids}}
                if where_clause:
                    combined = {"$and": [where_clause, cluster_where]}
                else:
                    combined = cluster_where
                where_clause = combined

        # 4. Dense retrieval — cosine similarity search
        n_dense = min(top_k * 5, total)
        dense_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results"       : n_dense,
            "include"         : ["documents", "metadatas", "distances"],
        }
        if where_clause:
            dense_kwargs["where"] = where_clause

        try:
            dense_results = collection.query(**dense_kwargs)
        except Exception as e:
            log.error(f"Dense retrieval error: {e}")
            return []

        # Parse dense results
        ids       = dense_results.get("ids", [[]])[0]
        docs      = dense_results.get("documents", [[]])[0]
        metas     = dense_results.get("metadatas", [[]])[0]
        distances = dense_results.get("distances", [[]])[0]

        # Convert distance to similarity (ChromaDB returns L2 or cosine distance)
        dense_scores = [max(0.0, 1.0 - d) for d in distances]

        # 5. Sparse retrieval — BM25 on retrieved candidates
        sparse_scores_map = {}
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            chunk_id = ids[i]
            bm25     = self._bm25_score(query_tokens, doc)
            sparse_scores_map[chunk_id] = bm25

        # 6. Normalize scores
        max_dense  = max(dense_scores, default=1.0) or 1.0
        max_sparse = max(sparse_scores_map.values(), default=1.0) or 1.0
        dense_norm  = [(ids[i], s / max_dense)  for i, s in enumerate(dense_scores)]
        sparse_norm = sorted(
            [(cid, s / max_sparse) for cid, s in sparse_scores_map.items()],
            key=lambda x: -x[1],
        )

        # 7. RRF fusion
        rrf_scores = self._rrf_fuse(dense_norm, sparse_norm)

        # 8. Build final results
        chunk_map = {ids[i]: (docs[i], metas[i], dense_scores[i])
                     for i in range(len(ids))}

        results = []
        for chunk_id, rrf_score in sorted(rrf_scores.items(),
                                           key=lambda x: -x[1]):
            if chunk_id not in chunk_map:
                continue
            doc, meta, d_score = chunk_map[chunk_id]
            s_score = sparse_scores_map.get(chunk_id, 0.0)

            results.append(VectorResult(
                chunk_id     = chunk_id,
                ticker       = meta.get("ticker", ""),
                filing_id    = meta.get("filing_id", ""),
                filing_type  = meta.get("filing_type", ""),
                filing_date  = meta.get("filing_date", ""),
                section_id   = meta.get("section_id", ""),
                section_name = meta.get("section_name", ""),
                text         = doc,
                dense_score  = round(d_score, 4),
                sparse_score = round(
                    sparse_scores_map.get(chunk_id, 0) / max_sparse, 4
                ),
                hybrid_score = round(rrf_score, 6),
                cluster_id   = int(meta.get("cluster_id", -1)),
                metadata     = {
                    "word_count"        : meta.get("word_count"),
                    "topic_shift_score" : meta.get("topic_shift_score"),
                    "chunk_index"       : meta.get("chunk_index"),
                    "total_chunks"      : meta.get("total_chunks"),
                    "overlap_previous"  : meta.get("overlap_previous", ""),
                    "overlap_next"      : meta.get("overlap_next", ""),
                },
            ))

        return results[:top_k]

    def _build_where(
        self,
        tickers      : Optional[list],
        sections     : Optional[list],
        filing_types : Optional[list],
        date_from    : Optional[str],
        date_to      : Optional[str],
    ) -> Optional[dict]:
        """Build ChromaDB where clause from filters."""
        conditions = []

        if tickers and len(tickers) == 1:
            conditions.append({"ticker": {"$eq": tickers[0].upper()}})
        elif tickers:
            conditions.append({"ticker": {"$in": [t.upper() for t in tickers]}})

        if sections and len(sections) == 1:
            conditions.append({"section_id": {"$eq": sections[0]}})
        elif sections:
            conditions.append({"section_id": {"$in": sections}})

        if filing_types and len(filing_types) == 1:
            conditions.append({"filing_type": {"$eq": filing_types[0]}})
        elif filing_types:
            conditions.append({"filing_type": {"$in": filing_types}})

        if date_from:
            conditions.append({"filing_date": {"$gte": date_from}})
        if date_to:
            conditions.append({"filing_date": {"$lte": date_to}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve_by_ticker(
        self,
        ticker  : str,
        query   : str,
        sections: Optional[list] = None,
        top_k   : int = 10,
    ) -> list[VectorResult]:
        """Convenience method: retrieve chunks for a specific ticker."""
        return self.retrieve(
            query    = query,
            tickers  = [ticker],
            sections = sections,
            top_k    = top_k,
        )

    def retrieve_cross_company(
        self,
        query   : str,
        section : str = "1A",
        top_k   : int = 20,
    ) -> list[VectorResult]:
        """
        Retrieve across ALL companies for a query.
        Useful for "which companies mention X" type questions.
        """
        return self.retrieve(
            query    = query,
            sections = [section],
            top_k    = top_k,
        )