# =============================================================================
# src/pipeline/graph_loader.py
# =============================================================================
# Step 5 of extraction pipeline.
# Reads data/extracted/TICKER/*.json
# Loads all entities, relations, and signals into Neo4j knowledge graph
#
# NODE TYPES:
#   Company           — ticker, name, sector
#   Person            — name, role, company
#   Risk              — category, subcategory, description
#   Event             — type, date, description
#   Market            — name, geography
#   Product           — name, company
#   Input             — commodity/component name
#   Geography         — country/region
#   FilingChunk       — chunk_id, filing_id, section, date
#
# RELATIONSHIP TYPES:
#   COMPETES_WITH     — Company → Company
#   SUPPLIES_TO       — Company → Company
#   BUYS_FROM         — Company → Company
#   DEPENDS_ON        — Company → Input
#   EXPOSED_TO        — Company → Risk
#   CAUSED            — Risk/Event → Risk/Event
#   IMPACTED_BY       — Company → Event
#   MENTIONED_IN      — Entity → FilingChunk (source tracing)
#   LED_BY            — Company → Person
#   OPERATES_IN       — Company → Market
#   HEDGES            — Company → Risk
#   LITIGATES_AGAINST — Company → Company/Regulator
#   PROPAGATES_TO     — Risk → Company (supply chain propagation)
#   HAS_CONCENTRATION — Company → Input/Company (concentration risk)
#
# Usage:
#   python src/pipeline/graph_loader.py            # all tickers
#   python src/pipeline/graph_loader.py AAPL MSFT  # specific tickers
#   python src/pipeline/graph_loader.py --clear    # wipe graph first

import sys
import json
import logging
import os
from pathlib import Path
from typing import Optional
from neo4j import GraphDatabase
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

EXTRACTED_DIR = Path("data/extracted")

NEO4J_HOST     = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT     = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]

# ---------------------------------------------------------------------------
# Neo4j driver
# ---------------------------------------------------------------------------
_driver = None

def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        uri = f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"
        log.info(f"Connecting to Neo4j at {uri}...")
        _driver = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
        _driver.verify_connectivity()
        log.info("Neo4j connected.")
    return _driver


def run_query(query: str, params: dict = None):
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return result.data()


# ---------------------------------------------------------------------------
# Schema setup — constraints and indexes for fast lookups
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
    "CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (r:Risk) REQUIRE r.risk_id IS UNIQUE",
    "CREATE CONSTRAINT input_id IF NOT EXISTS FOR (i:Input) REQUIRE i.input_id IS UNIQUE",
    "CREATE CONSTRAINT geography_id IF NOT EXISTS FOR (g:Geography) REQUIRE g.geo_id IS UNIQUE",
    "CREATE CONSTRAINT market_id IF NOT EXISTS FOR (m:Market) REQUIRE m.market_id IS UNIQUE",
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (f:FilingChunk) REQUIRE f.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
    "CREATE INDEX risk_category IF NOT EXISTS FOR (r:Risk) ON (r.category)",
    "CREATE INDEX chunk_ticker IF NOT EXISTS FOR (f:FilingChunk) ON (f.ticker)",
    "CREATE INDEX chunk_date IF NOT EXISTS FOR (f:FilingChunk) ON (f.filing_date)",
    "CREATE INDEX chunk_section IF NOT EXISTS FOR (f:FilingChunk) ON (f.section_id)",
]

def setup_schema():
    log.info("Setting up Neo4j schema...")
    for constraint in CONSTRAINTS:
        try:
            run_query(constraint)
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"Constraint: {e}")
    for index in INDEXES:
        try:
            run_query(index)
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"Index: {e}")
    log.info("Schema ready.")


def clear_graph():
    log.warning("Clearing entire graph...")
    run_query("MATCH (n) DETACH DELETE n")
    log.info("Graph cleared.")


# ---------------------------------------------------------------------------
# Node creation helpers — MERGE prevents duplicates
# ---------------------------------------------------------------------------

def upsert_company(ticker: str, name: str = None) -> str:
    """Create or update a Company node. Returns ticker."""
    if not ticker:
        return None
    run_query("""
        MERGE (c:Company {ticker: $ticker})
        ON CREATE SET c.name = $name, c.created_at = timestamp()
        ON MATCH SET c.name = COALESCE($name, c.name)
    """, {"ticker": ticker.upper(), "name": name or ticker.upper()})
    return ticker.upper()


def upsert_person(name: str, role: str, company: str) -> str:
    if not name:
        return None
    person_id = f"{name.lower().replace(' ', '_')}_{company}"
    run_query("""
        MERGE (p:Person {person_id: $person_id})
        ON CREATE SET p.name = $name, p.role = $role,
                      p.company = $company, p.created_at = timestamp()
        ON MATCH SET p.role = $role
    """, {"person_id": person_id, "name": name, "role": role, "company": company})
    return person_id


def upsert_risk(category: str, subcategory: str, description: str) -> str:
    if not category:
        return None
    risk_id = f"{category}_{subcategory or 'general'}".lower().replace(" ", "_")
    run_query("""
        MERGE (r:Risk {risk_id: $risk_id})
        ON CREATE SET r.category = $category, r.subcategory = $subcategory,
                      r.description = $description, r.created_at = timestamp()
        ON MATCH SET r.description = CASE WHEN $description IS NOT NULL
                     THEN $description ELSE r.description END
    """, {"risk_id": risk_id, "category": category,
          "subcategory": subcategory, "description": description})
    return risk_id


def upsert_input(input_name: str, input_type: str = None) -> str:
    if not input_name:
        return None
    input_id = input_name.lower().replace(" ", "_")
    run_query("""
        MERGE (i:Input {input_id: $input_id})
        ON CREATE SET i.name = $name, i.input_type = $input_type,
                      i.created_at = timestamp()
        ON MATCH SET i.input_type = COALESCE($input_type, i.input_type)
    """, {"input_id": input_id, "name": input_name, "input_type": input_type})
    return input_id


def upsert_geography(geo_name: str) -> str:
    if not geo_name:
        return None
    geo_id = geo_name.lower().replace(" ", "_")
    run_query("""
        MERGE (g:Geography {geo_id: $geo_id})
        ON CREATE SET g.name = $name, g.created_at = timestamp()
    """, {"geo_id": geo_id, "name": geo_name})
    return geo_id


def upsert_market(market_name: str) -> str:
    if not market_name:
        return None
    market_id = market_name.lower().replace(" ", "_")
    run_query("""
        MERGE (m:Market {market_id: $market_id})
        ON CREATE SET m.name = $name, m.created_at = timestamp()
    """, {"market_id": market_id, "name": market_name})
    return market_id


def upsert_chunk(chunk_data: dict) -> str:
    chunk_id = chunk_data.get("chunk_id")
    if not chunk_id:
        return None
    run_query("""
        MERGE (f:FilingChunk {chunk_id: $chunk_id})
        ON CREATE SET f.ticker = $ticker, f.filing_id = $filing_id,
                      f.filing_type = $filing_type, f.filing_date = $filing_date,
                      f.section_id = $section_id, f.section_name = $section_name,
                      f.word_count = $word_count, f.created_at = timestamp()
    """, {
        "chunk_id"    : chunk_id,
        "ticker"      : chunk_data.get("ticker"),
        "filing_id"   : chunk_data.get("filing_id"),
        "filing_type" : chunk_data.get("filing_type"),
        "filing_date" : chunk_data.get("filing_date"),
        "section_id"  : chunk_data.get("section_id"),
        "section_name": chunk_data.get("section_name"),
        "word_count"  : chunk_data.get("word_count"),
    })
    return chunk_id


def upsert_event(event_id: str, event_type: str, description: str,
                 geography: str = None, filing_date: str = None) -> str:
    if not event_id:
        return None
    run_query("""
        MERGE (e:Event {event_id: $event_id})
        ON CREATE SET e.event_type = $event_type, e.description = $description,
                      e.geography = $geography, e.filing_date = $filing_date,
                      e.created_at = timestamp()
    """, {"event_id": event_id, "event_type": event_type,
          "description": description, "geography": geography,
          "filing_date": filing_date})
    return event_id


# ---------------------------------------------------------------------------
# Relationship creation helpers
# ---------------------------------------------------------------------------

def create_relation(from_label: str, from_id_field: str, from_id: str,
                    to_label: str, to_id_field: str, to_id: str,
                    rel_type: str, props: dict = None):
    if not from_id or not to_id:
        return
    props = props or {}
    props_str = ", ".join(f"r.{k} = ${k}" for k in props)
    set_clause = f"SET {props_str}, r.updated_at = timestamp()" if props else "SET r.updated_at = timestamp()"

    query = f"""
        MATCH (a:{from_label} {{{from_id_field}: $from_id}})
        MATCH (b:{to_label} {{{to_id_field}: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
    """
    params = {"from_id": from_id, "to_id": to_id, **props}
    try:
        run_query(query, params)
    except Exception as e:
        log.debug(f"Relation {rel_type} {from_id}→{to_id}: {e}")


# ---------------------------------------------------------------------------
# Extraction loading — one chunk at a time
# ---------------------------------------------------------------------------

def load_chunk(chunk: dict, filing_date: str):
    """Load all extracted knowledge from one chunk into Neo4j."""
    ticker   = chunk.get("ticker", "").upper()
    chunk_id = chunk.get("chunk_id")
    ext      = chunk.get("extraction", {})

    if not ticker or not ext:
        return

    # Ensure the filing company node exists
    upsert_company(ticker)

    # Ensure the chunk node exists
    upsert_chunk(chunk)

    # Link company to its filing chunk
    create_relation("Company", "ticker", ticker,
                    "FilingChunk", "chunk_id", chunk_id,
                    "HAS_CHUNK", {"filing_date": filing_date})

    # ── Entities ──────────────────────────────────────────────────────────
    entities = ext.get("entities", {})

    # Companies mentioned
    for company in entities.get("companies", []):
        name   = company.get("name")
        ctick  = company.get("ticker", "").upper() if company.get("ticker") else None
        role   = company.get("role", "mentioned")
        ctx    = company.get("context", "")

        if not name:
            continue

        node_id = ctick if ctick else name.lower().replace(" ", "_")
        if ctick:
            upsert_company(ctick, name)
        else:
            # Create a company node without ticker
            run_query("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.name = $name, c.created_at = timestamp()
            """, {"ticker": node_id, "name": name})

        # Create relationship based on role
        rel_map = {
            "competitor"  : "COMPETES_WITH",
            "supplier"    : "SUPPLIES_TO",
            "customer"    : "BUYS_FROM",
            "partner"     : "PARTNERS_WITH",
            "acquirer"    : "ACQUIRED",
            "target"      : "ACQUIRED",
            "regulator"   : "REGULATES",
        }
        rel_type = rel_map.get(role)
        if rel_type and node_id != ticker:
            if role == "supplier":
                create_relation("Company", "ticker", node_id,
                                "Company", "ticker", ticker,
                                "SUPPLIES_TO",
                                {"context": ctx, "confidence": company.get("confidence", 0.7),
                                 "source_chunk": chunk_id, "filing_date": filing_date})
            elif role == "customer":
                create_relation("Company", "ticker", ticker,
                                "Company", "ticker", node_id,
                                "BUYS_FROM",
                                {"context": ctx, "source_chunk": chunk_id,
                                 "filing_date": filing_date})
            elif role == "competitor":
                create_relation("Company", "ticker", ticker,
                                "Company", "ticker", node_id,
                                "COMPETES_WITH",
                                {"context": ctx, "source_chunk": chunk_id,
                                 "filing_date": filing_date})
            elif role == "partner":
                create_relation("Company", "ticker", ticker,
                                "Company", "ticker", node_id,
                                "PARTNERS_WITH",
                                {"context": ctx, "source_chunk": chunk_id,
                                 "filing_date": filing_date})

        # MENTIONED_IN → chunk for source tracing
        create_relation("Company", "ticker", node_id,
                        "FilingChunk", "chunk_id", chunk_id,
                        "MENTIONED_IN", {"role": role, "filing_date": filing_date})

    # People
    for person in entities.get("people", []):
        name    = person.get("name")
        role    = person.get("role", "")
        company = person.get("company", ticker)
        if not name:
            continue
        pid = upsert_person(name, role, company)
        create_relation("Company", "ticker", ticker,
                        "Person", "person_id", pid,
                        "LED_BY",
                        {"role": role, "source_chunk": chunk_id,
                         "filing_date": filing_date})

    # Markets
    for market in entities.get("markets", []):
        mid = upsert_market(market)
        create_relation("Company", "ticker", ticker,
                        "Market", "market_id", mid,
                        "OPERATES_IN", {"source_chunk": chunk_id})

    # Geographies
    for geo in entities.get("geographies", []):
        gid = upsert_geography(geo)
        create_relation("Company", "ticker", ticker,
                        "Geography", "geo_id", gid,
                        "PRESENT_IN", {"source_chunk": chunk_id})

    # ── Risk signals ──────────────────────────────────────────────────────
    for risk in ext.get("risk_signals", []):
        cat    = risk.get("category")
        subcat = risk.get("subcategory", "general")
        desc   = risk.get("description", "")
        sev    = risk.get("severity", "medium")
        conf   = risk.get("confidence", 0.7)
        if not cat:
            continue
        rid = upsert_risk(cat, subcat, desc)
        create_relation("Company", "ticker", ticker,
                        "Risk", "risk_id", rid,
                        "EXPOSED_TO", {
                            "severity"       : sev,
                            "confidence"     : conf,
                            "forward_looking": risk.get("forward_looking", False),
                            "mitigation"     : risk.get("mitigation"),
                            "source_chunk"   : chunk_id,
                            "filing_date"    : filing_date,
                        })

    # ── Relations from extractor ──────────────────────────────────────────
    for rel in ext.get("relations", []):
        from_e = rel.get("from", "").upper()
        to_e   = rel.get("to", "").upper()
        rtype  = rel.get("relation", "RELATED_TO")
        conf   = rel.get("confidence", 0.7)
        mag    = rel.get("magnitude")
        seg    = rel.get("segment")

        if not from_e or not to_e:
            continue

        # Ensure both nodes exist as companies if they look like tickers
        if len(from_e) <= 6:
            upsert_company(from_e)
        if len(to_e) <= 6:
            upsert_company(to_e)

        try:
            create_relation("Company", "ticker", from_e,
                            "Company", "ticker", to_e,
                            rtype, {
                                "confidence"  : conf,
                                "magnitude"   : mag,
                                "segment"     : seg,
                                "source_chunk": chunk_id,
                                "filing_date" : filing_date,
                            })
        except Exception:
            pass  # May fail if to_e is a risk/event not a company

    # ── Causal chains ────────────────────────────────────────────────────
    for chain in ext.get("causal_chains", []):
        cause     = chain.get("cause", "")
        effect    = chain.get("effect", "")
        entity    = chain.get("affected_entity", ticker).upper()
        mechanism = chain.get("mechanism", "")
        conf      = chain.get("confidence", 0.7)
        timeframe = chain.get("timeframe")

        if not cause or not effect:
            continue

        cause_id  = f"cause_{cause[:50].lower().replace(' ', '_').replace(',', '')}"
        effect_id = f"effect_{effect[:50].lower().replace(' ', '_').replace(',', '')}"

        upsert_event(cause_id, "cause", cause, filing_date=filing_date)
        upsert_event(effect_id, "effect", effect, filing_date=filing_date)

        create_relation("Event", "event_id", cause_id,
                        "Event", "event_id", effect_id,
                        "CAUSED", {
                            "mechanism"   : mechanism,
                            "confidence"  : conf,
                            "timeframe"   : timeframe,
                            "source_chunk": chunk_id,
                            "filing_date" : filing_date,
                        })

        # Link affected entity to effect
        create_relation("Company", "ticker", entity,
                        "Event", "event_id", effect_id,
                        "IMPACTED_BY", {
                            "confidence"  : conf,
                            "source_chunk": chunk_id,
                            "filing_date" : filing_date,
                        })

    # ── Dependency chains ─────────────────────────────────────────────────
    for dep_chain in ext.get("dependency_chains", []):
        entity = dep_chain.get("ticker", ticker).upper() or ticker
        upsert_company(entity)

        for dep in dep_chain.get("depends_on", []):
            inp_name = dep.get("input")
            if not inp_name:
                continue
            inp_id = upsert_input(inp_name, dep.get("input_type"))

            create_relation("Company", "ticker", entity,
                            "Input", "input_id", inp_id,
                            "DEPENDS_ON", {
                                "criticality"      : dep.get("criticality", "medium"),
                                "substitutability" : dep.get("substitutability", "medium"),
                                "concentration_risk": dep.get("concentration_risk", False),
                                "cost_share"       : dep.get("cost_share"),
                                "confidence"       : dep.get("confidence", 0.7),
                                "source_chunk"     : chunk_id,
                                "filing_date"      : filing_date,
                            })

            # Link suppliers of this input
            for supplier in dep.get("suppliers_named", []):
                sup_id = supplier.upper().replace(" ", "_")
                upsert_company(sup_id, supplier)
                create_relation("Company", "ticker", sup_id,
                                "Input", "input_id", inp_id,
                                "PRODUCES", {
                                    "source_chunk": chunk_id,
                                    "filing_date" : filing_date,
                                })

            # Link supplier geographies
            for geo in dep.get("supplier_geographies", []):
                geo_id = upsert_geography(geo)
                create_relation("Input", "input_id", inp_id,
                                "Geography", "geo_id", geo_id,
                                "SOURCED_FROM", {
                                    "source_chunk": chunk_id,
                                    "filing_date" : filing_date,
                                })

    # ── Propagation risks ─────────────────────────────────────────────────
    for prop in ext.get("propagation_risks", []):
        trigger_type = prop.get("trigger_event_type", "unknown")
        trigger_geo  = prop.get("trigger_geography")
        input_aff    = prop.get("input_affected")
        conf         = prop.get("confidence", 0.7)

        # Create trigger event node
        event_id = (f"trigger_{trigger_type}_{trigger_geo or 'global'}"
                    .lower().replace(" ", "_"))
        upsert_event(event_id, trigger_type,
                     f"{trigger_type} in {trigger_geo or 'global'}",
                     geography=trigger_geo, filing_date=filing_date)

        # Link trigger to affected input
        if input_aff:
            inp_id = upsert_input(input_aff)
            create_relation("Event", "event_id", event_id,
                            "Input", "input_id", inp_id,
                            "DISRUPTS", {
                                "confidence"  : conf,
                                "source_chunk": chunk_id,
                                "filing_date" : filing_date,
                            })

        # First order impact
        first = prop.get("first_order_impact", {})
        first_ticker = (first.get("ticker") or "").upper()
        if first_ticker:
            upsert_company(first_ticker, first.get("entity"))
            create_relation("Event", "event_id", event_id,
                            "Company", "ticker", first_ticker,
                            "PROPAGATES_TO", {
                                "order"       : 1,
                                "impact_type" : first.get("impact_type"),
                                "severity"    : first.get("severity"),
                                "lag_time"    : first.get("lag_time"),
                                "confidence"  : first.get("confidence", conf),
                                "source_chunk": chunk_id,
                                "filing_date" : filing_date,
                            })

        # Second order impact
        second = prop.get("second_order_impact", {})
        second_ticker = (second.get("ticker") or "").upper()
        if second_ticker:
            upsert_company(second_ticker, second.get("entity"))
            create_relation("Event", "event_id", event_id,
                            "Company", "ticker", second_ticker,
                            "PROPAGATES_TO", {
                                "order"       : 2,
                                "impact_type" : second.get("impact_type"),
                                "severity"    : second.get("severity"),
                                "lag_time"    : second.get("lag_time"),
                                "confidence"  : second.get("confidence", conf * 0.8),
                                "source_chunk": chunk_id,
                                "filing_date" : filing_date,
                            })

    # ── Concentration disclosures ─────────────────────────────────────────
    for conc in ext.get("concentration_disclosures", []):
        ctype  = conc.get("type", "unknown")
        entity = conc.get("entity_named")
        pct    = conc.get("percentage")
        dep    = conc.get("dependency_level", "medium")
        conf   = conc.get("confidence", 0.7)
        risk   = conc.get("risk_if_lost", "")

        if entity:
            etick = (conc.get("ticker") or entity.upper()[:10]
                     .replace(" ", "_"))
            upsert_company(etick, entity)
            create_relation("Company", "ticker", ticker,
                            "Company", "ticker", etick,
                            "HAS_CONCENTRATION", {
                                "concentration_type" : ctype,
                                "percentage"         : pct,
                                "dependency_level"   : dep,
                                "risk_if_lost"       : risk,
                                "confidence"         : conf,
                                "source_chunk"       : chunk_id,
                                "filing_date"        : filing_date,
                            })

    # ── Hedging signals ───────────────────────────────────────────────────
    for hedge in ext.get("hedging_signals", []):
        risk_hedged = hedge.get("risk_being_hedged")
        specific    = hedge.get("specific_risk", "")
        conf        = hedge.get("confidence", 0.7)
        if not risk_hedged:
            continue
        rid = upsert_risk(f"hedged_{risk_hedged}", specific, specific)
        create_relation("Company", "ticker", ticker,
                        "Risk", "risk_id", rid,
                        "HEDGES", {
                            "instrument"  : hedge.get("instrument"),
                            "coverage_pct": hedge.get("coverage_percentage"),
                            "confidence"  : conf,
                            "source_chunk": chunk_id,
                            "filing_date" : filing_date,
                        })

    # ── Litigation signals ────────────────────────────────────────────────
    for lit in ext.get("litigation_signals", []):
        plaintiff = lit.get("plaintiff")
        defendant = lit.get("defendant")
        conf      = lit.get("confidence", 0.7)
        case_type = lit.get("case_type", "other")
        exposure  = lit.get("potential_exposure")

        if plaintiff and defendant:
            p_id = plaintiff.upper().replace(" ", "_")[:20]
            d_id = defendant.upper().replace(" ", "_")[:20]
            run_query("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.name = $name, c.created_at = timestamp()
            """, {"ticker": p_id, "name": plaintiff})
            run_query("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.name = $name, c.created_at = timestamp()
            """, {"ticker": d_id, "name": defendant})
            create_relation("Company", "ticker", p_id,
                            "Company", "ticker", d_id,
                            "LITIGATES_AGAINST", {
                                "case_type"   : case_type,
                                "exposure"    : exposure,
                                "stage"       : lit.get("stage"),
                                "confidence"  : conf,
                                "source_chunk": chunk_id,
                                "filing_date" : filing_date,
                            })

    # ── Geographic concentrations ─────────────────────────────────────────
    for geo_conc in ext.get("geographic_concentrations", []):
        location = geo_conc.get("concentrated_in")
        if not location:
            continue
        geo_id = upsert_geography(location)
        create_relation("Company", "ticker", ticker,
                        "Geography", "geo_id", geo_id,
                        "CONCENTRATED_IN", {
                            "concentration_type": geo_conc.get("concentration_type"),
                            "percentage"        : geo_conc.get("concentration_percentage"),
                            "risk_events"       : str(geo_conc.get("risk_events_applicable", [])),
                            "confidence"        : geo_conc.get("confidence", 0.7),
                            "source_chunk"      : chunk_id,
                            "filing_date"       : filing_date,
                        })


# ---------------------------------------------------------------------------
# File / ticker / main
# ---------------------------------------------------------------------------

def process_filing(extracted_path: Path) -> dict:
    data        = json.load(open(extracted_path))
    filing_id   = data.get("filing_id", "")
    filing_date = data.get("filing_date", "")
    chunks      = data.get("chunks", [])

    loaded = 0
    for chunk in chunks:
        if chunk.get("extraction_status") != "ok":
            continue
        try:
            load_chunk(chunk, filing_date)
            loaded += 1
        except Exception as e:
            log.error(f"  Chunk {chunk.get('chunk_id')}: {e}")

    return {"filing_id": filing_id, "loaded": loaded,
            "total": len(chunks)}


def process_ticker(ticker: str) -> dict:
    ext_dir = EXTRACTED_DIR / ticker
    if not ext_dir.exists():
        log.warning(f"{ticker}: no extracted directory — run extractor first")
        return {"ticker": ticker, "total_loaded": 0}

    files = sorted(ext_dir.glob("*.json"))
    if not files:
        return {"ticker": ticker, "total_loaded": 0}

    total_loaded = 0
    for f in files:
        try:
            r = process_filing(f)
            total_loaded += r["loaded"]
            log.info(f"  {f.name}: {r['loaded']}/{r['total']} chunks loaded")
        except Exception as e:
            log.error(f"  {f.name}: {e}")

    log.info(f"{ticker}: {total_loaded} chunks loaded into graph")
    return {"ticker": ticker, "total_loaded": total_loaded}


def main():
    args        = sys.argv[1:]
    do_clear    = "--clear" in args
    tickers_arg = [a for a in args if not a.startswith("--")]
    tickers     = tickers_arg if tickers_arg else TICKERS

    setup_schema()

    if do_clear:
        clear_graph()

    log.info(f"Loading {len(tickers)} tickers into Neo4j...")
    total_loaded = 0
    results = []
    for t in tickers:
        r = process_ticker(t)
        results.append(r)
        total_loaded += r["total_loaded"]

    # Graph stats
    node_count = run_query("MATCH (n) RETURN count(n) as count")[0]["count"]
    rel_count  = run_query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]

    print("\n" + "=" * 60)
    print("GRAPH LOADING COMPLETE")
    print("=" * 60)
    print(f"  Chunks loaded  : {total_loaded}")
    print(f"  Graph nodes    : {node_count}")
    print(f"  Graph relations: {rel_count}")
    print(f"  Neo4j Browser  : http://localhost:7474")
    print()
    for r in results:
        if r["total_loaded"] > 0:
            print(f"    {r['ticker']:<8} {r['total_loaded']} chunks")


if __name__ == "__main__":
    main()