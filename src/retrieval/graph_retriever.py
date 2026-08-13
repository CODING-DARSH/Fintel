# =============================================================================
# src/retrieval/graph_retriever.py
# =============================================================================
# Retrieves structured knowledge from Neo4j knowledge graph.
#
# CONNECTION (this fix): NEO4J_URI now takes priority over
# NEO4J_HOST/PORT when set — needed for AuraDB, which hands you a full
# "neo4j+s://xxxxxxxx.databases.neo4j.io" connection string (TLS, no
# separate host/port to assemble). Local/self-hosted Neo4j keeps
# working unchanged via the old host/port + plain "bolt://" path when
# NEO4J_URI is left unset.
#
# All other logic below (optional-arg fallbacks on every get_* method,
# node_exists, get_propagation_paths) is unchanged from the version
# already fixed elsewhere — only the connection block changed here.

import os
import re
import inspect
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

NEO4J_HOST     = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT     = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# NEO4J_URI takes priority when set — required for AuraDB. Local/self-hosted
# Neo4j keeps working unchanged via NEO4J_HOST/PORT when this is not set.
NEO4J_URI = os.getenv("NEO4J_URI", "")


@dataclass
class GraphResult:
    """Single result from graph retrieval."""
    result_type  : str
    primary_entity: str
    data         : dict
    score        : float = 1.0
    hop_distance : int   = 0
    source_chunks: list  = field(default_factory=list)
    filing_dates : list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "result_type"   : self.result_type,
            "primary_entity": self.primary_entity,
            "data"          : self.data,
            "score"         : self.score,
            "hop_distance"  : self.hop_distance,
            "source_chunks" : self.source_chunks,
            "filing_dates"  : self.filing_dates,
        }


class GraphRetriever:
    """
    Retrieves structured knowledge from Neo4j.
    Uses typed Cypher queries for each retrieval pattern.
    """

    def __init__(self):
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            uri = NEO4J_URI if NEO4J_URI else f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"
            self._driver = GraphDatabase.driver(
                uri,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
        return self._driver

    def _run(self, query: str, params: dict = None) -> list:
        try:
            driver = self._get_driver()
            with driver.session() as session:
                return session.run(query, params or {}).data()
        except Exception as e:
            log.error(f"Neo4j query error: {e}")
            return []

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    KNOWN_QUERY_METHODS_PREFIX = "get_"

    def query(self, query_type: str, **kwargs) -> list:
        if not query_type.startswith(self.KNOWN_QUERY_METHODS_PREFIX):
            log.warning(f"Rejected graph query_type (must start with 'get_'): {query_type!r}")
            return []

        method = getattr(self, query_type, None)
        if method is None or not callable(method):
            log.warning(f"Unknown graph query_type: {query_type!r}")
            return []

        if query_type == "get_company_overview":
            log.warning("get_company_overview returns a dict, not a list — use it directly instead of query()")
            return []

        sig = inspect.signature(method)
        valid_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters and v is not None
        }

        try:
            result = method(**valid_kwargs)
        except Exception as e:
            log.error(f"graph query {query_type!r} failed: {e}")
            return []

        return result if isinstance(result, list) else []

    def get_company_risks(
        self,
        ticker      : Optional[str] = None,
        severity    : Optional[str] = None,
        date_from   : Optional[str] = None,
        limit       : int = 20,
    ) -> list[GraphResult]:
        ticker_filter   = "AND c.ticker = $ticker" if ticker else ""
        severity_filter = "AND r.severity = $severity" if severity else ""
        date_filter     = "AND rel.filing_date >= $date_from" if date_from else ""

        query = f"""
            MATCH (c:Company)-[rel:EXPOSED_TO]->(r:Risk)
            WHERE 1=1 {ticker_filter} {severity_filter} {date_filter}
            OPTIONAL MATCH (r)<-[:EXPOSED_TO]-(other:Company)
            WITH c, r, rel, collect(DISTINCT other.ticker) AS shared_companies
            RETURN c.ticker      AS ticker,
                   r.risk_id      AS risk_id,
                   r.category     AS category,
                   r.subcategory  AS subcategory,
                   r.description  AS description,
                   rel.severity   AS severity,
                   rel.confidence AS confidence,
                   rel.filing_date AS filing_date,
                   rel.source_chunk AS source_chunk,
                   rel.mitigation  AS mitigation,
                   shared_companies
            ORDER BY rel.confidence DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        if severity:
            params["severity"] = severity
        if date_from:
            params["date_from"] = date_from

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "company_risk",
                primary_entity= row.get("ticker", ""),
                score         = float(row.get("confidence") or 0.8),
                hop_distance  = 1,
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                filing_dates  = [row["filing_date"]] if row.get("filing_date") else [],
                data={
                    "risk_id"         : row.get("risk_id"),
                    "category"        : row.get("category"),
                    "subcategory"     : row.get("subcategory"),
                    "description"     : row.get("description"),
                    "severity"        : row.get("severity"),
                    "mitigation"      : row.get("mitigation"),
                    "shared_with"     : row.get("shared_companies", []),
                },
            ))
        return results

    def get_shared_risk_companies(
        self,
        risk_category   : Optional[str] = None,
        risk_subcategory: Optional[str] = None,
        severity        : Optional[str] = None,
        limit           : int = 30,
    ) -> list[GraphResult]:
        cat_filter = "AND toLower(r.category) CONTAINS toLower($category)" if risk_category else ""
        sub_filter = ("AND toLower(r.subcategory) CONTAINS toLower($subcategory)"
                      if risk_subcategory else "")
        sev_filter = "AND rel.severity = $severity" if severity else ""

        query = f"""
            MATCH (c:Company)-[rel:EXPOSED_TO]->(r:Risk)
            WHERE 1=1 {cat_filter}
            {sub_filter} {sev_filter}
            RETURN c.ticker        AS ticker,
                   c.name          AS name,
                   r.category      AS category,
                   r.subcategory   AS subcategory,
                   r.description   AS description,
                   rel.severity    AS severity,
                   rel.confidence  AS confidence,
                   rel.filing_date AS filing_date,
                   rel.source_chunk AS source_chunk
            ORDER BY rel.confidence DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if risk_category:
            params["category"] = risk_category
        if risk_subcategory:
            params["subcategory"] = risk_subcategory
        if severity:
            params["severity"] = severity

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "shared_risk",
                primary_entity= row.get("ticker", ""),
                score         = float(row.get("confidence") or 0.8),
                hop_distance  = 1,
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                filing_dates  = [row["filing_date"]] if row.get("filing_date") else [],
                data={
                    "ticker"      : row.get("ticker"),
                    "name"        : row.get("name"),
                    "category"    : row.get("category"),
                    "subcategory" : row.get("subcategory"),
                    "description" : row.get("description"),
                    "severity"    : row.get("severity"),
                    "filing_date" : row.get("filing_date"),
                },
            ))
        return results

    def get_supply_chain(
        self,
        ticker   : Optional[str] = None,
        max_hops : int = 3,
        limit    : int = 50,
    ) -> list[GraphResult]:
        ticker_clause = "{ticker: $ticker}" if ticker else ""
        query = f"""
            MATCH path = (c:Company {ticker_clause})-[:DEPENDS_ON*1..{max_hops}]->(i:Input)
            WITH c, i, length(path) AS hops,
                 [r IN relationships(path) | r.criticality] AS criticalities,
                 [r IN relationships(path) | r.source_chunk] AS chunks
            OPTIONAL MATCH (supplier:Company)-[:PRODUCES]->(i)
            OPTIONAL MATCH (i)-[:SOURCED_FROM]->(geo:Geography)
            RETURN c.ticker        AS ticker,
                   i.name          AS input_name,
                   i.input_type    AS input_type,
                   hops            AS hop_distance,
                   criticalities[0] AS criticality,
                   collect(DISTINCT supplier.ticker) AS suppliers,
                   collect(DISTINCT geo.name)        AS geographies,
                   chunks[0]       AS source_chunk
            ORDER BY hops ASC, criticality DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()

        rows = self._run(query, params)
        results = []
        for row in rows:
            crit_map = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
            score = crit_map.get(row.get("criticality", "medium"), 0.6)
            score = score / max(row.get("hop_distance", 1), 1)

            results.append(GraphResult(
                result_type   = "supply_chain_dependency",
                primary_entity= row.get("ticker", ""),
                score         = score,
                hop_distance  = int(row.get("hop_distance", 1)),
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                data={
                    "input_name"   : row.get("input_name"),
                    "input_type"   : row.get("input_type"),
                    "criticality"  : row.get("criticality"),
                    "suppliers"    : row.get("suppliers", []),
                    "geographies"  : row.get("geographies", []),
                    "hop_distance" : row.get("hop_distance"),
                },
            ))
        return results

    def get_propagation_risks(
        self,
        ticker     : Optional[str] = None,
        event_type : Optional[str] = None,
        limit      : int = 20,
    ) -> list[GraphResult]:
        ticker_clause = "{ticker: $ticker}" if ticker else ""
        event_filter = ("AND toLower(e.event_type) CONTAINS toLower($event_type)"
                        if event_type else "")
        query = f"""
            MATCH (e:Event)-[rel:PROPAGATES_TO]->(c:Company {ticker_clause})
            WHERE 1=1 {event_filter}
            RETURN c.ticker      AS ticker,
                   e.event_id    AS event_id,
                   e.event_type  AS event_type,
                   e.description AS description,
                   e.geography   AS geography,
                   rel.order     AS propagation_order,
                   rel.impact_type AS impact_type,
                   rel.severity    AS severity,
                   rel.lag_time    AS lag_time,
                   rel.confidence  AS confidence,
                   rel.source_chunk AS source_chunk,
                   rel.filing_date  AS filing_date
            ORDER BY rel.confidence DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        if event_type:
            params["event_type"] = event_type

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "propagation_risk",
                primary_entity= row.get("ticker", ""),
                score         = float(row.get("confidence") or 0.7),
                hop_distance  = int(row.get("propagation_order") or 1),
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                filing_dates  = [row["filing_date"]] if row.get("filing_date") else [],
                data={
                    "event_type"        : row.get("event_type"),
                    "description"       : row.get("description"),
                    "geography"         : row.get("geography"),
                    "propagation_order" : row.get("propagation_order"),
                    "impact_type"       : row.get("impact_type"),
                    "severity"          : row.get("severity"),
                    "lag_time"          : row.get("lag_time"),
                },
            ))
        return results

    def get_competitors(
        self,
        ticker  : Optional[str] = None,
        segment : Optional[str] = None,
        limit   : int = 20,
    ) -> list[GraphResult]:
        ticker_clause = "{ticker: $ticker}" if ticker else ""
        seg_filter = ("AND toLower(rel.segment) CONTAINS toLower($segment)"
                      if segment else "")
        query = f"""
            MATCH (c:Company {ticker_clause})-[rel:COMPETES_WITH]->(comp:Company)
            WHERE 1=1 {seg_filter}
            RETURN comp.ticker     AS ticker,
                   comp.name       AS name,
                   rel.segment     AS segment,
                   rel.confidence  AS confidence,
                   rel.source_chunk AS source_chunk,
                   rel.filing_date  AS filing_date
            ORDER BY rel.confidence DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        if segment:
            params["segment"] = segment

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "competitor",
                primary_entity= row.get("ticker", ""),
                score         = float(row.get("confidence") or 0.8),
                hop_distance  = 1,
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                filing_dates  = [row["filing_date"]] if row.get("filing_date") else [],
                data={
                    "ticker"      : row.get("ticker"),
                    "name"        : row.get("name"),
                    "segment"     : row.get("segment"),
                    "filing_date" : row.get("filing_date"),
                },
            ))
        return results

    def get_causal_chains(
        self,
        cause_keyword: Optional[str] = None,
        limit        : int = 20,
    ) -> list[GraphResult]:
        keyword_filter = """
            WHERE toLower(cause.description) CONTAINS toLower($keyword)
               OR toLower(effect.description) CONTAINS toLower($keyword)
        """ if cause_keyword else ""
        query = f"""
            MATCH (cause:Event)-[rel:CAUSED]->(effect:Event)
            {keyword_filter}
            OPTIONAL MATCH (c:Company)-[:IMPACTED_BY]->(effect)
            RETURN cause.description  AS cause,
                   effect.description AS effect,
                   rel.mechanism      AS mechanism,
                   rel.confidence     AS confidence,
                   rel.timeframe      AS timeframe,
                   rel.source_chunk   AS source_chunk,
                   collect(DISTINCT c.ticker) AS affected_companies
            ORDER BY rel.confidence DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if cause_keyword:
            params["keyword"] = cause_keyword

        rows = self._run(query, params)
        results = []
        for row in rows:
            affected = row.get("affected_companies", [])
            primary = cause_keyword or (affected[0] if affected else "causal_chain")
            results.append(GraphResult(
                result_type   = "causal_chain",
                primary_entity= primary,
                score         = float(row.get("confidence") or 0.7),
                hop_distance  = 1,
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                data={
                    "cause"              : row.get("cause"),
                    "effect"             : row.get("effect"),
                    "mechanism"          : row.get("mechanism"),
                    "timeframe"          : row.get("timeframe"),
                    "affected_companies" : affected,
                },
            ))
        return results

    def get_macro_impact(
        self,
        indicator : Optional[str] = None,
        magnitude : Optional[str] = None,
        limit     : int = 20,
    ) -> list[GraphResult]:
        ind_filter = "AND toLower(m.indicator) CONTAINS toLower($indicator)" if indicator else ""
        mag_filter = "AND m.magnitude = $magnitude" if magnitude else ""
        query = f"""
            MATCH (m:MacroSignal)-[rel:AFFECTS]->(c:Company)
            WHERE 1=1 {ind_filter} {mag_filter}
            RETURN c.ticker     AS ticker,
                   c.name       AS name,
                   m.indicator  AS indicator,
                   m.direction  AS direction,
                   m.magnitude  AS magnitude,
                   m.value      AS value,
                   m.fetch_date AS fetch_date,
                   rel.magnitude AS impact_magnitude
            ORDER BY rel.magnitude DESC, m.fetch_date DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if indicator:
            params["indicator"] = indicator
        if magnitude:
            params["magnitude"] = magnitude

        rows = self._run(query, params)
        results = []
        for row in rows:
            mag_map = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
            score = mag_map.get(row.get("magnitude", "medium"), 0.6)
            results.append(GraphResult(
                result_type   = "macro_impact",
                primary_entity= row.get("ticker", ""),
                score         = score,
                hop_distance  = 1,
                data={
                    "ticker"           : row.get("ticker"),
                    "name"             : row.get("name"),
                    "indicator"        : row.get("indicator"),
                    "direction"        : row.get("direction"),
                    "magnitude"        : row.get("magnitude"),
                    "value"            : row.get("value"),
                    "fetch_date"       : row.get("fetch_date"),
                    "impact_magnitude" : row.get("impact_magnitude"),
                },
            ))
        return results

    def get_news_impact(
        self,
        ticker   : Optional[str] = None,
        urgency  : Optional[str] = None,
        days_back: int = 30,
        limit    : int = 20,
    ) -> list[GraphResult]:
        ticker_clause = "{ticker: $ticker}" if ticker else ""
        urg_filter = "AND n.urgency = $urgency" if urgency else ""
        query = f"""
            MATCH (n:NewsArticle)-[rel:MENTIONS]->(c:Company {ticker_clause})
            WHERE 1=1 {urg_filter}
            RETURN c.ticker        AS ticker,
                   n.article_id    AS article_id,
                   n.title         AS title,
                   n.summary       AS summary,
                   n.event_type    AS event_type,
                   n.urgency       AS urgency,
                   n.published_date AS published_date,
                   n.url           AS url,
                   rel.sentiment   AS sentiment,
                   rel.role        AS role
            ORDER BY n.published_date DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        if urgency:
            params["urgency"] = urgency

        rows = self._run(query, params)
        results = []
        for row in rows:
            urg_map = {"breaking": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
            score = urg_map.get(row.get("urgency", "medium"), 0.6)
            results.append(GraphResult(
                result_type   = "news_mention",
                primary_entity= row.get("ticker", ""),
                score         = score,
                hop_distance  = 1,
                data={
                    "article_id"    : row.get("article_id"),
                    "title"         : row.get("title"),
                    "summary"       : row.get("summary"),
                    "event_type"    : row.get("event_type"),
                    "urgency"       : row.get("urgency"),
                    "published_date": row.get("published_date"),
                    "url"           : row.get("url"),
                    "sentiment"     : row.get("sentiment"),
                    "role"          : row.get("role"),
                },
            ))
        return results

    def get_executive_changes(
        self,
        ticker   : Optional[str] = None,
        role     : Optional[str] = None,
        limit    : int = 20,
    ) -> list[GraphResult]:
        ticker_filter = "WHERE c.ticker = $ticker" if ticker else "WHERE 1=1"
        role_filter   = "AND toLower(p.role) CONTAINS toLower($role)" if role else ""

        query = f"""
            MATCH (c:Company)-[rel:LED_BY]->(p:Person)
            {ticker_filter} {role_filter}
            RETURN c.ticker    AS ticker,
                   p.name      AS name,
                   p.role      AS role,
                   rel.filing_date AS filing_date,
                   rel.source_chunk AS source_chunk
            ORDER BY rel.filing_date DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        if role:
            params["role"] = role

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "executive_change",
                primary_entity= row.get("ticker", ""),
                score         = 0.9,
                hop_distance  = 1,
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                filing_dates  = [row["filing_date"]] if row.get("filing_date") else [],
                data={
                    "ticker"      : row.get("ticker"),
                    "person_name" : row.get("name"),
                    "role"        : row.get("role"),
                    "filing_date" : row.get("filing_date"),
                },
            ))
        return results

    def get_geographic_exposure(
        self,
        geography: Optional[str] = None,
        limit    : int = 20,
    ) -> list[GraphResult]:
        geo_filter = "WHERE toLower(g.name) CONTAINS toLower($geography)" if geography else ""
        query = f"""
            MATCH (c:Company)-[rel:CONCENTRATED_IN|PRESENT_IN]->(g:Geography)
            {geo_filter}
            RETURN c.ticker              AS ticker,
                   c.name               AS name,
                   g.name               AS geography,
                   type(rel)            AS relation_type,
                   rel.concentration_type AS concentration_type,
                   rel.percentage        AS percentage,
                   rel.source_chunk      AS source_chunk
            ORDER BY COALESCE(rel.percentage, -1) DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if geography:
            params["geography"] = geography

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "geographic_exposure",
                primary_entity= row.get("ticker", ""),
                score         = 0.85,
                hop_distance  = 1,
                source_chunks = [row["source_chunk"]] if row.get("source_chunk") else [],
                data={
                    "ticker"            : row.get("ticker"),
                    "name"              : row.get("name"),
                    "geography"         : row.get("geography"),
                    "relation_type"     : row.get("relation_type"),
                    "concentration_type": row.get("concentration_type"),
                    "percentage"        : row.get("percentage"),
                },
            ))
        return results

    def get_market_signals(
        self,
        ticker      : Optional[str] = None,
        signal_type : Optional[str] = None,
        limit       : int = 10,
    ) -> list[GraphResult]:
        ticker_clause = "{ticker: $ticker}" if ticker else ""
        sig_filter = "AND s.type = $signal_type" if signal_type else ""
        query = f"""
            MATCH (c:Company {ticker_clause})-[rel:HAS_MARKET_SIGNAL]->(s:MarketSignal)
            WHERE 1=1 {sig_filter}
            RETURN c.ticker      AS ticker,
                   s.type       AS signal_type,
                   s.note       AS note,
                   s.fetch_date AS fetch_date,
                   c.latest_price   AS latest_price,
                   c.pct_change_1d  AS pct_change_1d,
                   c.volume_ratio   AS volume_ratio,
                   c.above_ma200    AS above_ma200
            ORDER BY s.fetch_date DESC
            LIMIT $limit
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        if signal_type:
            params["signal_type"] = signal_type

        rows = self._run(query, params)
        results = []
        for row in rows:
            results.append(GraphResult(
                result_type   = "market_signal",
                primary_entity= row.get("ticker", ""),
                score         = 0.9,
                hop_distance  = 0,
                data={
                    "signal_type"  : row.get("signal_type"),
                    "note"         : row.get("note"),
                    "fetch_date"   : row.get("fetch_date"),
                    "latest_price" : row.get("latest_price"),
                    "pct_change_1d": row.get("pct_change_1d"),
                    "volume_ratio" : row.get("volume_ratio"),
                    "above_ma200"  : row.get("above_ma200"),
                },
            ))
        return results

    def get_company_overview(self, ticker: str) -> dict:
        query = """
            MATCH (c:Company {ticker: $ticker})
            OPTIONAL MATCH (c)-[:EXPOSED_TO]->(r:Risk)
            OPTIONAL MATCH (c)-[:COMPETES_WITH]->(comp:Company)
            OPTIONAL MATCH (c)-[:DEPENDS_ON]->(i:Input)
            OPTIONAL MATCH (c)-[:OPERATES_IN]->(m:Market)
            OPTIONAL MATCH (c)-[:LED_BY]->(p:Person)
            RETURN c.ticker         AS ticker,
                   c.name           AS name,
                   c.latest_price   AS latest_price,
                   c.pct_change_1d  AS pct_change_1d,
                   c.above_ma200    AS above_ma200,
                   c.market_updated_at AS market_updated_at,
                   collect(DISTINCT {
                     category: r.category,
                     severity: r.severity
                   }) AS risks,
                   collect(DISTINCT comp.ticker) AS competitors,
                   collect(DISTINCT i.name)      AS dependencies,
                   collect(DISTINCT m.name)      AS markets,
                   collect(DISTINCT {
                     name: p.name, role: p.role
                   }) AS executives
        """
        rows = self._run(query, {"ticker": ticker.upper()})
        return rows[0] if rows else {}

    PROPAGATION_REL_TYPES = (
        "DEPENDS_ON|SUPPLIES_TO|SOURCED_FROM|EXPOSED_TO|PROPAGATES_TO|BUYS_FROM"
        "|CONCENTRATED_IN|HAS_CONCENTRATION"
    )

    START_LABEL_ID_FIELD = {
        "Input"      : "input_id",
        "Geography"  : "geo_id",
        "Event"      : "event_id",
        "Company"    : "ticker",
    }

    def node_exists(self, label: str, id_value: str) -> bool:
        if label not in self.START_LABEL_ID_FIELD:
            return False
        id_field = self.START_LABEL_ID_FIELD[label]
        query = f"MATCH (n:{label} {{{id_field}: $id_value}}) RETURN count(n) AS c LIMIT 1"
        try:
            rows = self._run(query, {"id_value": id_value})
        except Exception as e:
            log.warning(f"node_exists check failed for {label}:{id_value}: {e}")
            return False
        return bool(rows and rows[0].get("c", 0) > 0)

    def get_propagation_paths(
        self,
        start_label : str,
        start_id    : str,
        max_depth   : int = 4,
        path_limit  : int = 50,
    ) -> list[dict]:
        if start_label not in self.START_LABEL_ID_FIELD:
            log.warning(f"get_propagation_paths: unknown start_label {start_label!r}")
            return []
        if max_depth < 1 or max_depth > 6:
            log.warning(f"get_propagation_paths: max_depth {max_depth} out of sane range, clamping to 1-6")
            max_depth = max(1, min(6, max_depth))

        id_field = self.START_LABEL_ID_FIELD[start_label]

        query = f"""
            MATCH (start:{start_label} {{{id_field}: $start_id}})
            MATCH p = (start)-[rels:{self.PROPAGATION_REL_TYPES}*1..{max_depth}]-(end:Company)
            WHERE start <> end
            WITH p, end, length(p) AS depth
            ORDER BY depth ASC
            LIMIT $path_limit
            RETURN
                end.ticker AS ticker,
                depth,
                [n IN nodes(p) | {{
                    labels: labels(n),
                    id: coalesce(n.ticker, n.input_id, n.geo_id, n.event_id),
                    name: coalesce(n.name, n.ticker, n.description)
                }}] AS path_nodes,
                [r IN relationships(p) | {{
                    type: type(r),
                    start_id: coalesce(startNode(r).ticker, startNode(r).input_id,
                                        startNode(r).geo_id, startNode(r).event_id),
                    end_id: coalesce(endNode(r).ticker, endNode(r).input_id,
                                      endNode(r).geo_id, endNode(r).event_id),
                    confidence: r.confidence,
                    criticality: r.criticality,
                    severity: r.severity,
                    dependency_level: r.dependency_level,
                    cost_share: r.cost_share,
                    percentage: r.percentage
                }}] AS path_rels
        """

        try:
            rows = self._run(query, {"start_id": start_id, "path_limit": path_limit})
        except Exception as e:
            log.error(f"get_propagation_paths query failed: {e}")
            return []

        if len(rows) >= path_limit:
            log.warning(
                f"get_propagation_paths hit path_limit ({path_limit}) for "
                f"start={start_label}:{start_id} — results may be truncated. "
                f"Consider a smaller max_depth or more specific start node."
            )

        return rows