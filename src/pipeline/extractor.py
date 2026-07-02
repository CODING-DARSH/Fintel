# =============================================================================
# src/pipeline/extractor.py  — v3
# =============================================================================
# Full extraction schema — everything extractable from filing TEXT prose.
#
# Added over v2:
#   risk_language_signals      — intensity markers, certainty level,
#                                hedging language, new vs repeated disclosure
#   concentration_disclosures  — supplier/customer/geography concentration
#                                with % dependency and risk-if-lost
#   hedging_signals            — what risks company is hedging reveals
#                                what they're most worried about
#   litigation_signals         — named parties, exposure amounts, stage,
#                                regulatory investigations
#   segment_performance        — management's own explanation of each
#                                business segment performance
#   debt_covenant_signals      — covenant triggers, headroom, breach risk
#   off_balance_sheet          — operating leases, SPVs, guarantees,
#                                contingent liabilities
#   management_tone            — language patterns revealing confidence
#                                or defensiveness in management commentary

import sys
import json
import logging
import os
import time
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_DIR    = Path("data/chunks")
EXTRACTED_DIR = Path("data/extracted")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

REQUESTS_PER_SECOND = 3
RETRY_ATTEMPTS      = 3
RETRY_DELAY         = 20

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
_groq = None

def get_groq():
    global _groq
    if _groq is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        from groq import Groq
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial intelligence extraction engine specialized
in supply chain analysis, risk propagation, dependency mapping, and financial
signal extraction from SEC/regulatory filing text.

Rules:
- Extract ONLY what is explicitly stated or strongly implied in the text
- Never hallucinate entities, numbers, or relationships not present
- Confidence: 0.9+ explicitly stated, 0.7-0.9 strongly implied,
  0.5-0.7 reasonably inferred, below 0.5 speculative
- For risk language: extract the EXACT phrasing that reveals intensity
- For concentrations: extract even partial info (% without name is still useful)
- For litigation: extract all named parties and dollar amounts mentioned
- Return ONLY valid JSON. No explanation, no markdown, no backticks."""


def build_prompt(chunk: dict) -> str:
    ticker      = chunk["ticker"]
    filing_type = chunk["filing_type"]
    filing_date = chunk["filing_date"]
    section     = chunk["section_name"]
    text        = chunk["text"]
    overlap     = chunk.get("overlap_previous", "")
    preceding   = chunk.get("preceding_context", "")

    context = ""
    if preceding:
        context += f"[Context from prior section: {preceding}]\n"
    if overlap:
        context += f"[Overlap from previous chunk: {overlap}]\n"

    return f"""Extract comprehensive structured financial intelligence from this filing chunk.

Company: {ticker}
Filing: {filing_type} dated {filing_date}
Section: {section}
{context}

TEXT TO ANALYZE:
{text}

Return a JSON object with EXACTLY this structure (null for missing fields,
[] for missing lists, never omit a key):

{{
  "entities": {{
    "companies": [
      {{
        "name": "string",
        "ticker": "string or null",
        "role": "subject|competitor|supplier|customer|regulator|partner|acquirer|target",
        "context": "string"
      }}
    ],
    "people": [
      {{
        "name": "string",
        "role": "string",
        "company": "string",
        "context": "string"
      }}
    ],
    "products": ["string"],
    "markets": ["string"],
    "regulators": ["string"],
    "geographies": ["string"]
  }},

  "financial_signals": [
    {{
      "type": "revenue_growth|revenue_decline|margin_expansion|margin_compression|debt_increase|debt_reduction|guidance_raised|guidance_lowered|beat_earnings|missed_earnings|cost_increase|cost_reduction|cash_flow_improvement|cash_flow_decline",
      "metric": "string",
      "direction": "up|down|flat",
      "magnitude": "low|medium|high",
      "value": "string or null",
      "cause": "string or null",
      "timeframe": "string or null",
      "confidence": 0.0
    }}
  ],

  "risk_signals": [
    {{
      "category": "macro_risk|competition_risk|regulatory_risk|supply_chain_risk|execution_risk|litigation_risk|cyber_risk|esg_risk|financial_risk|geopolitical_risk|talent_risk|technology_risk",
      "subcategory": "string",
      "severity": "low|medium|high|critical",
      "description": "string",
      "affected_segments": ["string"],
      "mitigation_mentioned": true,
      "mitigation": "string or null",
      "forward_looking": true,
      "confidence": 0.0
    }}
  ],

  "risk_language_signals": [
    {{
      "risk_topic": "string (what risk this statement is about)",
      "exact_phrase": "string (the exact words used, max 50 words)",
      "intensity_markers": ["string (e.g. materially, significantly, will, may, could)"],
      "intensity_level": "low|medium|high|critical",
      "certainty_level": "possible|probable|likely|certain",
      "hedging_language_used": true,
      "is_quantified": true,
      "quantification": "string or null (e.g. up to $500M exposure)",
      "forward_looking": true,
      "note": "string or null (any notable language pattern worth flagging)",
      "confidence": 0.0
    }}
  ],

  "concentration_disclosures": [
    {{
      "type": "supplier|customer|geography|product|channel|employee|technology",
      "entity_named": "string or null",
      "ticker": "string or null",
      "percentage": "string or null (e.g. 30% of revenue, 25% of COGS)",
      "absolute_value": "string or null (e.g. $2.3B)",
      "dependency_level": "low|medium|high|critical",
      "risk_if_lost": "string (what happens if this concentration fails)",
      "mitigation": "string or null",
      "trend": "increasing|stable|decreasing|unknown",
      "confidence": 0.0
    }}
  ],

  "hedging_signals": [
    {{
      "risk_being_hedged": "currency|commodity|interest_rate|credit|equity|other",
      "specific_risk": "string (e.g. USD/EUR exchange rate, oil price)",
      "instrument": "futures|options|swaps|forward_contracts|insurance|natural_hedge|other",
      "coverage_percentage": "string or null",
      "notional_value": "string or null",
      "duration": "string or null",
      "implication": "string (what hedging this risk reveals about management priority)",
      "confidence": 0.0
    }}
  ],

  "litigation_signals": [
    {{
      "case_type": "patent|antitrust|employment|regulatory|environmental|securities|consumer|contract|other",
      "plaintiff": "string or null",
      "defendant": "string or null",
      "regulator_involved": "string or null (SEC, DOJ, FTC, SEBI, CCI, etc.)",
      "description": "string",
      "claimed_amount": "string or null (e.g. $500M, unspecified)",
      "potential_exposure": "string or null",
      "stage": "investigation|filed|discovery|trial|appeal|settled|dismissed|ongoing",
      "management_assessment": "string or null (how mgmt characterizes it)",
      "confidence": 0.0
    }}
  ],

  "segment_performance": [
    {{
      "segment_name": "string",
      "direction": "growth|decline|flat|mixed",
      "magnitude": "low|medium|high",
      "revenue_mentioned": "string or null",
      "margin_mentioned": "string or null",
      "management_explanation": "string (mgmt's own words on why)",
      "key_drivers": ["string"],
      "key_headwinds": ["string"],
      "outlook": "string or null",
      "confidence": 0.0
    }}
  ],

  "debt_covenant_signals": [
    {{
      "covenant_type": "financial_ratio|leverage|interest_coverage|liquidity|other",
      "description": "string",
      "threshold": "string or null (e.g. debt/EBITDA must stay below 3.5x)",
      "current_headroom": "string or null",
      "breach_risk": "low|medium|high|critical",
      "consequence_if_breached": "string or null",
      "lender_named": "string or null",
      "confidence": 0.0
    }}
  ],

  "off_balance_sheet": [
    {{
      "type": "operating_lease|guarantee|contingent_liability|SPV|joint_venture|take_or_pay|purchase_commitment|other",
      "description": "string",
      "estimated_value": "string or null",
      "timeframe": "string or null",
      "risk_level": "low|medium|high",
      "trigger_condition": "string or null (what activates this liability)",
      "confidence": 0.0
    }}
  ],

  "management_tone": {{
    "overall_tone": "confident|cautious|defensive|optimistic|pessimistic|neutral|mixed",
    "tone_indicators": ["string (specific phrases that signal the tone)"],
    "deflection_detected": true,
    "deflection_examples": ["string or null"],
    "blame_externalized": true,
    "blame_targets": ["string (macro, competitors, government, etc.)"],
    "forward_confidence": "high|medium|low",
    "unusual_language": "string or null (anything notably unusual in wording)",
    "confidence": 0.0
  }},

  "strategic_signals": [
    {{
      "type": "market_expansion|market_exit|acquisition|divestiture|partnership|product_launch|restructuring|leadership_change|capital_allocation|new_business_model",
      "description": "string",
      "markets": ["string"],
      "companies_involved": ["string"],
      "stage": "announced|in_progress|completed|planned",
      "financial_impact": "string or null",
      "confidence": 0.0
    }}
  ],

  "causal_chains": [
    {{
      "cause": "string",
      "effect": "string",
      "affected_entity": "string",
      "mechanism": "string",
      "timeframe": "string or null",
      "confidence": 0.0
    }}
  ],

  "relations": [
    {{
      "from": "string",
      "relation": "COMPETES_WITH|SUPPLIES_TO|BUYS_FROM|PARTNERS_WITH|EXPOSED_TO|CAUSED|IMPACTED_BY|REGULATES|OWNS|DIVESTED|ACQUIRED|DEPENDS_ON|HEDGES|LITIGATES_AGAINST",
      "to": "string",
      "segment": "string or null",
      "magnitude": "low|medium|high or null",
      "timeframe": "string or null",
      "confidence": 0.0
    }}
  ],

  "dependency_chains": [
    {{
      "entity": "string",
      "ticker": "string or null",
      "depends_on": [
        {{
          "input": "string",
          "input_type": "commodity|component|energy|labor|service|technology|capital",
          "suppliers_named": ["string"],
          "supplier_geographies": ["string"],
          "criticality": "low|medium|high|critical",
          "substitutability": "low|medium|high",
          "concentration_risk": true,
          "cost_share": "string or null",
          "confidence": 0.0
        }}
      ]
    }}
  ],

  "propagation_risks": [
    {{
      "trigger_event_type": "mine_disruption|port_congestion|trade_policy_change|labor_strike|natural_disaster|sanctions|currency_move|energy_shock|regulatory_change|geopolitical_event|pandemic|logistics_disruption",
      "trigger_geography": "string",
      "input_affected": "string",
      "first_order_impact": {{
        "entity": "string",
        "ticker": "string or null",
        "impact_type": "cost_increase|supply_shortage|production_halt|delivery_delay|revenue_loss",
        "severity": "low|medium|high|critical",
        "lag_time": "string",
        "confidence": 0.0
      }},
      "second_order_impact": {{
        "entity": "string",
        "ticker": "string or null",
        "impact_type": "cost_increase|supply_shortage|production_halt|delivery_delay|margin_compression",
        "severity": "low|medium|high",
        "lag_time": "string",
        "confidence": 0.0
      }},
      "historical_precedent": "string or null",
      "confidence": 0.0
    }}
  ],

  "input_cost_sensitivity": [
    {{
      "input": "string",
      "cost_share": "string or null",
      "price_passthrough": "low|medium|high|full",
      "passthrough_lag": "string or null",
      "historical_pattern": "string or null",
      "hedging_mentioned": true,
      "hedging_description": "string or null",
      "confidence": 0.0
    }}
  ],

  "geographic_concentrations": [
    {{
      "entity": "string",
      "concentrated_in": "string",
      "concentration_type": "production|sourcing|revenue|workforce|regulatory",
      "concentration_percentage": "string or null",
      "risk_events_applicable": ["string"],
      "mitigation": "string or null",
      "confidence": 0.0
    }}
  ],

  "forward_looking_statements": [
    {{
      "type": "guidance|outlook|projection|strategy|warning|commitment",
      "direction": "positive|negative|neutral|mixed",
      "metric": "string or null",
      "description": "string",
      "timeframe": "string or null",
      "confidence": 0.0
    }}
  ],

  "chunk_metadata": {{
    "topics": ["string"],
    "primary_theme": "string",
    "has_numbers": true,
    "has_guidance": true,
    "has_competitor_mentions": true,
    "has_risk_disclosure": true,
    "has_causal_language": true,
    "has_supply_chain_content": true,
    "has_dependency_disclosure": true,
    "has_litigation": true,
    "has_hedging_disclosure": true,
    "has_segment_breakdown": true,
    "has_covenant_disclosure": true,
    "has_concentration_disclosure": true,
    "information_density": "low|medium|high",
    "filing_specific_context": "string or null"
  }}
}}

Return ONLY the JSON object. No explanation, no markdown, no backticks."""


# ---------------------------------------------------------------------------
# Groq call with retry
# ---------------------------------------------------------------------------

def call_groq(prompt: str, attempt: int = 0) -> Optional[dict]:
    client = get_groq()
    try:
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw.strip())

    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error (attempt {attempt+1}): {e}")
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(2)
            return call_groq(prompt, attempt + 1)
        return None

    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            if attempt < RETRY_ATTEMPTS - 1:
                log.warning(f"Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                return call_groq(prompt, attempt + 1)
        log.error(f"Groq error: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-chunk extraction
# ---------------------------------------------------------------------------

def extract_chunk(chunk: dict) -> dict:
    prompt     = build_prompt(chunk)
    extraction = call_groq(prompt)
    return {
        "chunk_id"          : chunk["chunk_id"],
        "ticker"            : chunk["ticker"],
        "filing_id"         : chunk["filing_id"],
        "filing_type"       : chunk["filing_type"],
        "filing_date"       : chunk["filing_date"],
        "section_id"        : chunk["section_id"],
        "section_name"      : chunk["section_name"],
        "chunk_index"       : chunk["chunk_index"],
        "total_chunks"      : chunk["total_chunks"],
        "word_count"        : chunk["word_count"],
        "text"              : chunk["text"],
        "extraction_status" : "ok" if extraction else "failed",
        "extraction"        : extraction or {},
    }


# ---------------------------------------------------------------------------
# File / ticker / main
# ---------------------------------------------------------------------------

def process_filing(chunk_path: Path, out_dir: Path, resume: bool) -> dict:
    out_path = out_dir / chunk_path.name
    if resume and out_path.exists():
        existing = json.load(open(out_path))
        log.info(f"  SKIP: {chunk_path.name}")
        return {"filing_id": existing.get("filing_id"),
                "chunks": len(existing.get("chunks", [])),
                "failed": 0, "status": "skipped"}

    data   = json.load(open(chunk_path))
    chunks = data.get("chunks", [])
    if not chunks:
        return {"filing_id": data.get("filing_id"),
                "chunks": 0, "failed": 0, "status": "empty"}

    extracted, failed = [], 0
    for i, chunk in enumerate(chunks):
        log.info(f"    [{i+1}/{len(chunks)}] {chunk['chunk_id']}")
        result = extract_chunk(chunk)
        extracted.append(result)
        if result["extraction_status"] == "failed":
            failed += 1
        time.sleep(1.0 / REQUESTS_PER_SECOND)

    out_path.write_text(json.dumps({
        "filing_id"          : data["filing_id"],
        "ticker"             : data["ticker"],
        "filing_type"        : data["filing_type"],
        "filing_date"        : data["filing_date"],
        "total_chunks"       : len(extracted),
        "failed_extractions" : failed,
        "chunks"             : extracted,
    }, indent=2, ensure_ascii=False))

    return {"filing_id": data["filing_id"], "chunks": len(extracted),
            "failed": failed, "status": "ok"}


def process_ticker(ticker: str, resume: bool) -> dict:
    chunk_dir = CHUNKS_DIR    / ticker
    out_dir   = EXTRACTED_DIR / ticker
    if not chunk_dir.exists():
        log.warning(f"{ticker}: no chunks — run chunker first")
        return {"ticker": ticker, "total_chunks": 0, "total_failed": 0}

    files = sorted(chunk_dir.glob("*.json"))
    if not files:
        return {"ticker": ticker, "total_chunks": 0, "total_failed": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    total_chunks, total_failed = 0, 0
    for f in files:
        try:
            r = process_filing(f, out_dir, resume)
            total_chunks += r["chunks"]
            total_failed += r["failed"]
        except Exception as e:
            log.error(f"  {f.name}: {e}")

    log.info(f"{ticker}: {total_chunks} chunks, {total_failed} failed")
    return {"ticker": ticker, "total_chunks": total_chunks,
            "total_failed": total_failed}


def main():
    args        = sys.argv[1:]
    resume      = "--resume" in args
    tickers_arg = [a for a in args if not a.startswith("--")]
    tickers     = tickers_arg if tickers_arg else TICKERS

    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set.")
        sys.exit(1)

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting {len(tickers)} tickers (resume={resume})...")

    total_chunks, total_failed, results = 0, 0, []
    for t in tickers:
        r = process_ticker(t, resume)
        results.append(r)
        total_chunks += r["total_chunks"]
        total_failed += r["total_failed"]

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Failed          : {total_failed}")
    print(f"  Success rate    : "
          f"{((total_chunks-total_failed)/max(total_chunks,1)*100):.1f}%")
    print(f"  Output          : {EXTRACTED_DIR.resolve()}")
    for r in results:
        if r["total_chunks"] > 0:
            print(f"    {r['ticker']:<8} {r['total_chunks']} chunks, "
                  f"{r['total_failed']} failed")


if __name__ == "__main__":
    main()