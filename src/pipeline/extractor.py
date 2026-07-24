# =============================================================================
# src/pipeline/extractor.py  — v6
# =============================================================================
# Two providers only: Gemini (google-genai) + Groq
# Up to 3 keys each, round-robin rotation across all keys
# Per-chunk save (crash-safe), chunk-level --resume
#
# Add to .env:
#   GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3
#   GROQ_API_KEY_1,   GROQ_API_KEY_2,   GROQ_API_KEY_3
#
# Usage:
#   python src/pipeline/extractor.py
#   python src/pipeline/extractor.py AAPL
#   python src/pipeline/extractor.py --resume
#   python src/pipeline/extractor.py --all-sections
#   python src/pipeline/extractor.py AAPL --resume

import sys
import json
import logging
import os
import time
import re
import threading
from pathlib import Path
from typing import Optional
from itertools import cycle

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_DIR    = Path("data/chunks")
EXTRACTED_DIR = Path("data/extracted")

HIGH_VALUE_SECTIONS = {
    "1A", "7", "7A",
    "2.02", "5.02", "1.01", "8.01", "7.01",
}

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]

RETRY_DELAY   = 25
REQUEST_DELAY = 1.0
MAX_ATTEMPTS  = 6


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def _load_keys(prefix):
    keys = []
    for i in range(1, 6):
        k = os.getenv(f"{prefix}_{i}", "").strip()
        if k:
            keys.append(k)
    return keys


GEMINI_KEYS = _load_keys("GEMINI_API_KEY")
GROQ_KEYS   = _load_keys("GROQ_API_KEY")

# Interleave: G1 → Gr1 → G2 → Gr2 → G3 → Gr3
_rotation_list = []
for i in range(max(len(GEMINI_KEYS), len(GROQ_KEYS), 1)):
    if i < len(GEMINI_KEYS):
        _rotation_list.append(("gemini", GEMINI_KEYS[i]))
    if i < len(GROQ_KEYS):
        _rotation_list.append(("groq",   GROQ_KEYS[i]))

if not _rotation_list:
    log.error("No API keys found. Add GEMINI_API_KEY_1 or GROQ_API_KEY_1 to .env")
    sys.exit(1)

_key_cycle  = cycle(_rotation_list)
_cycle_lock = threading.Lock()

def get_next():
    with _cycle_lock:
        return next(_key_cycle)

log.info(f"Keys: Gemini={len(GEMINI_KEYS)} Groq={len(GROQ_KEYS)} "
         f"Total rotation={len(_rotation_list)}")


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def _call_gemini(key: str, prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a financial intelligence extraction engine. "
                "Return ONLY valid JSON. No explanation, no markdown, no backticks."
            ),
            temperature=0.1,
            max_output_tokens=8000,
        ),
    )
    return response.text.strip()


def _call_groq(key: str, prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial intelligence extraction engine. "
                    "Return ONLY valid JSON. No explanation, no markdown, no backticks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=8000,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


CALLERS = {"gemini": _call_gemini, "groq": _call_groq}


# ---------------------------------------------------------------------------
# Unified call with rotation + retry
# ---------------------------------------------------------------------------

def call_llm(prompt: str, attempt: int = 0) -> Optional[dict]:
    if attempt >= MAX_ATTEMPTS:
        log.error("Max attempts reached — skipping chunk")
        return None

    provider, key = get_next()
    try:
        raw = CALLERS[provider](key, prompt)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw.strip())

    except json.JSONDecodeError as e:
        log.warning(f"[{provider}] JSON error attempt {attempt+1}: {str(e)[:60]}")
        time.sleep(2)
        return call_llm(prompt, attempt + 1)

    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ["429","quota","rate","limit","exceeded","too many"]):
            log.warning(f"[{provider}] Rate limit — waiting {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        else:
            log.error(f"[{provider}] Error: {str(e)[:100]}")
        return call_llm(prompt, attempt + 1)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(chunk: dict) -> str:
    context = ""
    if chunk.get("preceding_context"):
        context += f"[Context from prior section: {chunk['preceding_context']}]\n"
    if chunk.get("overlap_previous"):
        context += f"[Overlap from previous chunk: {chunk['overlap_previous']}]\n"

    return f"""Extract comprehensive structured financial intelligence.

Company: {chunk['ticker']}
Filing: {chunk['filing_type']} dated {chunk['filing_date']}
Section: {chunk['section_name']}
{context}

TEXT TO ANALYZE:
{chunk['text']}

Return a JSON object with EXACTLY this structure
(null for missing fields, [] for missing lists, never omit a key):

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
      {{"name": "string", "role": "string", "company": "string", "context": "string"}}
    ],
    "products": ["string"],
    "markets": ["string"],
    "regulators": ["string"],
    "geographies": ["string"]
  }},
  "financial_signals": [
    {{
      "type": "revenue_growth|revenue_decline|margin_expansion|margin_compression|debt_increase|debt_reduction|guidance_raised|guidance_lowered|beat_earnings|missed_earnings|cost_increase|cost_reduction",
      "metric": "string", "direction": "up|down|flat", "magnitude": "low|medium|high",
      "value": "string or null", "cause": "string or null",
      "timeframe": "string or null", "confidence": 0.0
    }}
  ],
  "risk_signals": [
    {{
      "category": "macro_risk|competition_risk|regulatory_risk|supply_chain_risk|execution_risk|litigation_risk|cyber_risk|esg_risk|financial_risk|geopolitical_risk|talent_risk|technology_risk",
      "subcategory": "string", "severity": "low|medium|high|critical",
      "description": "string", "affected_segments": ["string"],
      "mitigation_mentioned": true, "mitigation": "string or null",
      "forward_looking": true, "confidence": 0.0
    }}
  ],
  "risk_language_signals": [
    {{
      "risk_topic": "string", "exact_phrase": "string (max 50 words)",
      "intensity_markers": ["string"], "intensity_level": "low|medium|high|critical",
      "certainty_level": "possible|probable|likely|certain",
      "hedging_language_used": true, "is_quantified": true,
      "quantification": "string or null", "forward_looking": true, "confidence": 0.0
    }}
  ],
  "concentration_disclosures": [
    {{
      "type": "supplier|customer|geography|product|channel|employee|technology",
      "entity_named": "string or null", "ticker": "string or null",
      "percentage": "string or null", "absolute_value": "string or null",
      "dependency_level": "low|medium|high|critical", "risk_if_lost": "string",
      "mitigation": "string or null", "trend": "increasing|stable|decreasing|unknown",
      "confidence": 0.0
    }}
  ],
  "hedging_signals": [
    {{
      "risk_being_hedged": "currency|commodity|interest_rate|credit|equity|other",
      "specific_risk": "string",
      "instrument": "futures|options|swaps|forward_contracts|insurance|natural_hedge|other",
      "coverage_percentage": "string or null", "notional_value": "string or null",
      "implication": "string", "confidence": 0.0
    }}
  ],
  "litigation_signals": [
    {{
      "case_type": "patent|antitrust|employment|regulatory|environmental|securities|consumer|contract|other",
      "plaintiff": "string or null", "defendant": "string or null",
      "regulator_involved": "string or null", "description": "string",
      "claimed_amount": "string or null", "potential_exposure": "string or null",
      "stage": "investigation|filed|discovery|trial|appeal|settled|dismissed|ongoing",
      "management_assessment": "string or null", "confidence": 0.0
    }}
  ],
  "segment_performance": [
    {{
      "segment_name": "string", "direction": "growth|decline|flat|mixed",
      "magnitude": "low|medium|high", "revenue_mentioned": "string or null",
      "margin_mentioned": "string or null", "management_explanation": "string",
      "key_drivers": ["string"], "key_headwinds": ["string"],
      "outlook": "string or null", "confidence": 0.0
    }}
  ],
  "debt_covenant_signals": [
    {{
      "covenant_type": "financial_ratio|leverage|interest_coverage|liquidity|other",
      "description": "string", "threshold": "string or null",
      "current_headroom": "string or null", "breach_risk": "low|medium|high|critical",
      "consequence_if_breached": "string or null", "confidence": 0.0
    }}
  ],
  "off_balance_sheet": [
    {{
      "type": "operating_lease|guarantee|contingent_liability|SPV|joint_venture|take_or_pay|purchase_commitment|other",
      "description": "string", "estimated_value": "string or null",
      "timeframe": "string or null", "risk_level": "low|medium|high",
      "trigger_condition": "string or null", "confidence": 0.0
    }}
  ],
  "management_tone": {{
    "overall_tone": "confident|cautious|defensive|optimistic|pessimistic|neutral|mixed",
    "tone_indicators": ["string"], "deflection_detected": true,
    "blame_externalized": true, "blame_targets": ["string"],
    "forward_confidence": "high|medium|low",
    "unusual_language": "string or null", "confidence": 0.0
  }},
  "strategic_signals": [
    {{
      "type": "market_expansion|market_exit|acquisition|divestiture|partnership|product_launch|restructuring|leadership_change|capital_allocation|new_business_model",
      "description": "string", "markets": ["string"],
      "companies_involved": ["string"],
      "stage": "announced|in_progress|completed|planned",
      "financial_impact": "string or null", "confidence": 0.0
    }}
  ],
  "causal_chains": [
    {{
      "cause": "string", "effect": "string", "affected_entity": "string",
      "mechanism": "string", "timeframe": "string or null", "confidence": 0.0
    }}
  ],
  "relations": [
    {{
      "from": "string",
      "relation": "COMPETES_WITH|SUPPLIES_TO|BUYS_FROM|PARTNERS_WITH|EXPOSED_TO|CAUSED|IMPACTED_BY|REGULATES|OWNS|DIVESTED|ACQUIRED|DEPENDS_ON|HEDGES|LITIGATES_AGAINST",
      "to": "string", "segment": "string or null",
      "magnitude": "low|medium|high or null",
      "timeframe": "string or null", "confidence": 0.0
    }}
  ],
  "dependency_chains": [
    {{
      "entity": "string", "ticker": "string or null",
      "depends_on": [
        {{
          "input": "string",
          "input_type": "commodity|component|energy|labor|service|technology|capital",
          "suppliers_named": ["string"], "supplier_geographies": ["string"],
          "criticality": "low|medium|high|critical",
          "substitutability": "low|medium|high",
          "concentration_risk": true, "cost_share": "string or null",
          "confidence": 0.0
        }}
      ]
    }}
  ],
  "propagation_risks": [
    {{
      "trigger_event_type": "mine_disruption|port_congestion|trade_policy_change|labor_strike|natural_disaster|sanctions|currency_move|energy_shock|regulatory_change|geopolitical_event|pandemic|logistics_disruption",
      "trigger_geography": "string", "input_affected": "string",
      "first_order_impact": {{
        "entity": "string", "ticker": "string or null",
        "impact_type": "cost_increase|supply_shortage|production_halt|delivery_delay|revenue_loss",
        "severity": "low|medium|high|critical", "lag_time": "string", "confidence": 0.0
      }},
      "second_order_impact": {{
        "entity": "string", "ticker": "string or null",
        "impact_type": "cost_increase|supply_shortage|production_halt|delivery_delay|margin_compression",
        "severity": "low|medium|high", "lag_time": "string", "confidence": 0.0
      }},
      "historical_precedent": "string or null", "confidence": 0.0
    }}
  ],
  "input_cost_sensitivity": [
    {{
      "input": "string", "cost_share": "string or null",
      "price_passthrough": "low|medium|high|full",
      "passthrough_lag": "string or null", "historical_pattern": "string or null",
      "hedging_mentioned": true, "hedging_description": "string or null",
      "confidence": 0.0
    }}
  ],
  "geographic_concentrations": [
    {{
      "entity": "string", "concentrated_in": "string",
      "concentration_type": "production|sourcing|revenue|workforce|regulatory",
      "concentration_percentage": "string or null",
      "risk_events_applicable": ["string"],
      "mitigation": "string or null", "confidence": 0.0
    }}
  ],
  "forward_looking_statements": [
    {{
      "type": "guidance|outlook|projection|strategy|warning|commitment",
      "direction": "positive|negative|neutral|mixed",
      "metric": "string or null", "description": "string",
      "timeframe": "string or null", "confidence": 0.0
    }}
  ],
  "chunk_metadata": {{
    "topics": ["string"], "primary_theme": "string",
    "has_numbers": true, "has_guidance": true,
    "has_competitor_mentions": true, "has_risk_disclosure": true,
    "has_causal_language": true, "has_supply_chain_content": true,
    "has_dependency_disclosure": true, "has_litigation": true,
    "has_hedging_disclosure": true, "has_segment_breakdown": true,
    "has_covenant_disclosure": true, "has_concentration_disclosure": true,
    "information_density": "low|medium|high",
    "filing_specific_context": "string or null"
  }}
}}

Return ONLY the JSON object. No explanation, no markdown, no backticks."""


# ---------------------------------------------------------------------------
# Per-chunk crash-safe save
# ---------------------------------------------------------------------------

def _load_output(out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.load(open(out_path))
        except Exception:
            pass
    return {"chunks": []}


def save_chunk(out_path: Path, result: dict, meta: dict):
    existing = _load_output(out_path)
    chunks   = existing.get("chunks", [])
    ids      = [c["chunk_id"] for c in chunks]
    if result["chunk_id"] in ids:
        chunks[ids.index(result["chunk_id"])] = result
    else:
        chunks.append(result)
    failed = sum(1 for c in chunks if c.get("extraction_status") == "failed")
    out_path.write_text(json.dumps(
        {**meta, "total_chunks": len(chunks),
         "failed_extractions": failed, "chunks": chunks},
        indent=2, ensure_ascii=False
    ))


def done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    try:
        data = json.load(open(out_path))
        return {c["chunk_id"] for c in data.get("chunks", [])
                if c.get("extraction_status") == "ok"}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_filing(chunk_path, out_dir, resume, all_sections):
    out_path = out_dir / chunk_path.name
    data     = json.load(open(chunk_path))
    chunks   = data.get("chunks", [])
    if not chunks:
        return {"filing_id": data.get("filing_id"), "chunks": 0, "failed": 0}

    if not all_sections:
        chunks = [c for c in chunks if c.get("section_id") in HIGH_VALUE_SECTIONS]
    if not chunks:
        return {"filing_id": data.get("filing_id"), "chunks": 0, "failed": 0}

    meta = {k: data[k] for k in
            ("filing_id", "ticker", "filing_type", "filing_date")}

    done    = done_ids(out_path) if resume else set()
    pending = [c for c in chunks if c["chunk_id"] not in done]

    if resume and done:
        log.info(f"    Resume: {len(done)} done, {len(pending)} remaining")

    if not pending:
        return {"filing_id": data["filing_id"],
                "chunks": len(chunks), "failed": 0}

    extracted = failed = 0
    for i, chunk in enumerate(pending):
        provider = _rotation_list[(i + len(done)) % len(_rotation_list)][0]
        log.info(f"    [{len(done)+i+1}/{len(chunks)}] "
                 f"{chunk['chunk_id']} → {provider}")

        ext = call_llm(build_prompt(chunk))
        result = {
            "chunk_id"         : chunk["chunk_id"],
            "ticker"           : chunk["ticker"],
            "filing_id"        : chunk["filing_id"],
            "filing_type"      : chunk["filing_type"],
            "filing_date"      : chunk["filing_date"],
            "section_id"       : chunk["section_id"],
            "section_name"     : chunk["section_name"],
            "chunk_index"      : chunk["chunk_index"],
            "total_chunks"     : chunk["total_chunks"],
            "word_count"       : chunk["word_count"],
            "text"             : chunk["text"],
            "extraction_status": "ok" if ext else "failed",
            "extraction"       : ext or {},
        }
        save_chunk(out_path, result, meta)   # crash-safe immediate save
        if ext:
            extracted += 1
        else:
            failed += 1
        time.sleep(REQUEST_DELAY)

    return {"filing_id": data["filing_id"],
            "chunks": extracted + len(done), "failed": failed}


def process_ticker(ticker, resume, all_sections):
    chunk_dir = CHUNKS_DIR    / ticker
    out_dir   = EXTRACTED_DIR / ticker
    if not chunk_dir.exists():
        log.warning(f"{ticker}: no chunks — run chunker first")
        return {"ticker": ticker, "total_chunks": 0, "total_failed": 0}
    files = sorted(chunk_dir.glob("*.json"))
    if not files:
        return {"ticker": ticker, "total_chunks": 0, "total_failed": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = tf = 0
    for f in files:
        try:
            r = process_filing(f, out_dir, resume, all_sections)
            tc += r["chunks"]; tf += r["failed"]
            log.info(f"  {f.name}: {r['chunks']} chunks, {r['failed']} failed")
        except Exception as e:
            log.error(f"  {f.name}: {e}")
    return {"ticker": ticker, "total_chunks": tc, "total_failed": tf}


def main():
    args         = sys.argv[1:]
    resume       = "--resume"       in args
    all_sections = "--all-sections" in args
    tickers_arg  = [a for a in args if not a.startswith("--")]
    tickers      = tickers_arg or TICKERS

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    tc = tf = 0
    results = []
    for t in tickers:
        r = process_ticker(t, resume, all_sections)
        results.append(r); tc += r["total_chunks"]; tf += r["total_failed"]

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Gemini keys : {len(GEMINI_KEYS)}")
    print(f"  Groq keys   : {len(GROQ_KEYS)}")
    print(f"  Total keys  : {len(_rotation_list)}")
    print(f"  Chunks      : {tc}")
    print(f"  Failed      : {tf}")
    print(f"  Success     : {((tc-tf)/max(tc,1)*100):.1f}%")
    for r in results:
        if r["total_chunks"] > 0:
            print(f"    {r['ticker']:<8} {r['total_chunks']} chunks "
                  f"{r['total_failed']} failed")


if __name__ == "__main__":
    main()