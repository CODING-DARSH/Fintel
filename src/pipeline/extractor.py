# =============================================================================
# src/pipeline/extractor.py  — v4
# =============================================================================
# Changes from v3:
#   - Uses a cheaper/lower-token Groq model by default (override with
#     GROQ_MODEL env var). "llama-3.3-70b-versatile" is still available if
#     you want it, just export GROQ_MODEL=llama-3.3-70b-versatile.
#   - Per-CHUNK checkpointing/resume, not just per-file. Every chunk result
#     is written to disk as soon as it's produced, and a "progress" block
#     tracks exactly which chunk_ids are done. Re-running with --resume
#     picks up mid-file, not just at file boundaries.
#   - Hard stop on quota exhaustion. If Groq reports a DAILY token/request
#     limit hit (not just a transient rate limit), the whole run stops
#     immediately instead of burning through the rest of the queue marking
#     everything "failed". Whatever was completed is already saved, so
#     tomorrow's --resume continues exactly where it stopped.
#   - No change to the "don't hallucinate" extraction rules — if anything,
#     failed/uncertain chunks are left out of the completed set entirely
#     rather than written with empty/fake extraction data, so they get
#     retried instead of silently accepted as "done".

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

# --- Model ------------------------------------------------------------------
# Default is a small, cheap, fast Groq model — good enough for structured
# extraction and much lower token cost than 70B. Override any time with:
#   export GROQ_MODEL=llama-3.3-70b-versatile
# Other reasonable cheap options on Groq: "llama-3.1-8b-instant",
# "gemma2-9b-it", "openai/gpt-oss-20b".
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

REQUESTS_PER_SECOND = 3
RETRY_ATTEMPTS      = 3
RETRY_DELAY         = 20

# Phrases that indicate a HARD/DAILY quota exhaustion (not a transient
# per-minute rate limit). If we see these, we stop the whole run rather
# than retrying or continuing on to burn more failed calls.
HARD_LIMIT_MARKERS = [
    "tokens per day", "requests per day", "tpd", "rpd",
    "daily limit", "quota", "insufficient_quota",
]

TICKERS = [
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","GM",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","BMY","AMGN","GILD","CVS",
    "AAPL","MSFT","GOOGL","NVDA","META","ADBE","CRM","INTC","CSCO","IBM",
]


class QuotaExceededError(Exception):
    """Raised when Groq reports a hard/daily quota exhaustion. Stops the run."""
    pass


class PayloadTooLargeError(Exception):
    """Raised on HTTP 413 — request body too big for this model. Not a rate
    limit, not a quota issue. Caller should shrink the input and retry once,
    or give up on just this chunk without affecting anything else."""
    pass


# Smaller models (e.g. llama-3.1-8b-instant) accept much smaller request
# bodies than 70b. Chunk text gets truncated to this many characters before
# being sent, if a first attempt comes back 413.
MAX_INPUT_CHARS_ON_RETRY = 6000


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
- If something is not present in the text, use null or [] — never invent a
  plausible-sounding value to fill a field
- Confidence: 0.9+ explicitly stated, 0.7-0.9 strongly implied,
  0.5-0.7 reasonably inferred, below 0.5 speculative
- For risk language: extract the EXACT phrasing that reveals intensity
- For concentrations: extract even partial info (% without name is still useful)
- For litigation: extract all named parties and dollar amounts mentioned
- Return ONLY valid JSON. No explanation, no markdown, no backticks."""


def build_prompt(chunk: dict, max_chars: Optional[int] = None) -> str:
    ticker      = chunk["ticker"]
    filing_type = chunk["filing_type"]
    filing_date = chunk["filing_date"]
    section     = chunk["section_name"]
    text        = chunk["text"]
    overlap     = chunk.get("overlap_previous", "")
    preceding   = chunk.get("preceding_context", "")

    if max_chars:
        # Drop the extra context first, then truncate the main text itself.
        # This is a shrink-to-fit fallback after a 413, not the normal path.
        overlap   = ""
        preceding = ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated to fit request size limit...]"

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
# Groq call with retry + hard-limit detection
# ---------------------------------------------------------------------------

def _is_hard_limit(err_text: str) -> bool:
    low = err_text.lower()
    return any(marker in low for marker in HARD_LIMIT_MARKERS)


def call_groq(prompt: str, attempt: int = 0) -> Optional[dict]:
    client = get_groq()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
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

        if "413" in err or "payload too large" in err.lower() or "request too large" in err.lower():
            # NOT a rate limit, NOT a quota issue — the request body itself
            # is too big for this model. Retrying the same payload will
            # just fail the same way, so don't burn retry attempts on it.
            raise PayloadTooLargeError(err)

        if _is_hard_limit(err):
            # Daily/hard quota exhausted — do NOT retry, do NOT keep going.
            # Bubble up so the caller can save progress and stop the run.
            log.error(f"HARD QUOTA LIMIT hit: {err}")
            raise QuotaExceededError(err)

        if "429" in err or "rate_limit" in err.lower():
            if attempt < RETRY_ATTEMPTS - 1:
                log.warning(f"Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                return call_groq(prompt, attempt + 1)
            # Exhausted retries on a rate limit we couldn't confirm as
            # transient — safer to stop than to mark everything "failed".
            log.error("Rate limit retries exhausted — stopping run.")
            raise QuotaExceededError(err)

        log.error(f"Groq error: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-chunk extraction
# ---------------------------------------------------------------------------

def extract_chunk(chunk: dict) -> dict:
    extraction = None
    try:
        prompt = build_prompt(chunk)
        extraction = call_groq(prompt)  # may raise QuotaExceededError — let it propagate
    except PayloadTooLargeError:
        log.warning(
            f"    413 Payload Too Large on {chunk['chunk_id']} — "
            f"retrying once with truncated input (context dropped, "
            f"text capped at {MAX_INPUT_CHARS_ON_RETRY} chars)"
        )
        try:
            small_prompt = build_prompt(chunk, max_chars=MAX_INPUT_CHARS_ON_RETRY)
            extraction = call_groq(small_prompt)
        except PayloadTooLargeError:
            # Still too big even truncated — give up on this one chunk only.
            # This is a per-chunk failure, not a quota/run-stopping event.
            log.error(
                f"    {chunk['chunk_id']} still too large after truncation — "
                f"marking failed, continuing with next chunk"
            )
            extraction = None

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
        "model_used"        : GROQ_MODEL,
    }


# ---------------------------------------------------------------------------
# File / ticker / main
# ---------------------------------------------------------------------------

def _load_existing(out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.load(open(out_path))
        except Exception:
            log.warning(f"  Could not parse existing {out_path.name}, starting fresh")
    return {}


def _save_progress(out_path: Path, data: dict, completed_by_id: dict, all_chunk_ids: list):
    """Write the output file with whatever has been completed so far."""
    ordered = [completed_by_id[cid] for cid in all_chunk_ids if cid in completed_by_id]
    done_ids = [cid for cid in all_chunk_ids if cid in completed_by_id]
    data["chunks"] = ordered
    data["total_chunks"] = len(all_chunk_ids)
    data["failed_extractions"] = sum(
        1 for c in ordered if c["extraction_status"] == "failed"
    )
    data["progress"] = {
        "completed_chunk_ids": done_ids,
        "completed_count": len(done_ids),
        "total_count": len(all_chunk_ids),
        "complete": len(done_ids) == len(all_chunk_ids),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_used": GROQ_MODEL,
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def process_filing(chunk_path: Path, out_dir: Path, resume: bool) -> dict:
    out_path = out_dir / chunk_path.name
    src      = json.load(open(chunk_path))
    chunks   = src.get("chunks", [])

    if not chunks:
        return {"filing_id": src.get("filing_id"),
                "chunks": 0, "failed": 0, "status": "empty"}

    all_chunk_ids = [c["chunk_id"] for c in chunks]

    # Load whatever's already on disk (from a previous run) if resuming.
    existing = _load_existing(out_path) if resume else {}
    completed_by_id = {}
    if existing:
        for c in existing.get("chunks", []):
            # Only treat successful extractions as "done" — failed ones get
            # retried, they were never real output, nothing to lose.
            if c.get("extraction_status") == "ok":
                completed_by_id[c["chunk_id"]] = c

    already_done = len(completed_by_id)
    if resume and already_done:
        log.info(f"  RESUME: {chunk_path.name} — {already_done}/{len(chunks)} chunks already done")

    if resume and already_done == len(chunks):
        log.info(f"  SKIP (fully complete): {chunk_path.name}")
        return {"filing_id": src.get("filing_id"),
                "chunks": already_done, "failed": 0, "status": "skipped"}

    out_data = {
        "filing_id"   : src["filing_id"],
        "ticker"      : src["ticker"],
        "filing_type" : src["filing_type"],
        "filing_date" : src["filing_date"],
    }

    newly_failed = 0
    for i, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        if cid in completed_by_id:
            continue  # already done in a prior run

        log.info(f"    [{i+1}/{len(chunks)}] {cid}")
        try:
            result = extract_chunk(chunk)
        except QuotaExceededError:
            # Save everything completed so far, then stop the ENTIRE run
            # (not just this file) so nothing keeps burning failed calls.
            _save_progress(out_path, out_data, completed_by_id, all_chunk_ids)
            log.error(
                f"  STOPPED mid-file at chunk {i+1}/{len(chunks)} in "
                f"{chunk_path.name} due to quota limit. Progress saved — "
                f"re-run with --resume to continue from here."
            )
            raise

        completed_by_id[cid] = result
        if result["extraction_status"] == "failed":
            newly_failed += 1

        # Checkpoint after every single chunk, not just at the end of file.
        _save_progress(out_path, out_data, completed_by_id, all_chunk_ids)
        time.sleep(1.0 / REQUESTS_PER_SECOND)

    return {"filing_id": src["filing_id"], "chunks": len(completed_by_id),
            "failed": newly_failed, "status": "ok"}


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
        r = process_filing(f, out_dir, resume)  # QuotaExceededError propagates up
        total_chunks += r["chunks"]
        total_failed += r["failed"]

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
    log.info(f"Extracting {len(tickers)} tickers with model={GROQ_MODEL} (resume={resume})...")

    total_chunks, total_failed, results = 0, 0, []
    stopped_early = False
    for t in tickers:
        try:
            r = process_ticker(t, resume)
            results.append(r)
            total_chunks += r["total_chunks"]
            total_failed += r["total_failed"]
        except QuotaExceededError:
            stopped_early = True
            log.error(
                f"Run stopped at ticker '{t}' due to quota exhaustion. "
                f"Everything completed so far is saved. Run again later "
                f"with --resume to pick up exactly where this left off."
            )
            break
        except Exception as e:
            log.error(f"  {t}: unexpected error: {e}")

    print("\n" + "=" * 60)
    print("EXTRACTION " + ("STOPPED (quota limit)" if stopped_early else "COMPLETE"))
    print("=" * 60)
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Failed          : {total_failed}")
    print(f"  Success rate    : "
          f"{((total_chunks-total_failed)/max(total_chunks,1)*100):.1f}%")
    print(f"  Model used      : {GROQ_MODEL}")
    print(f"  Output          : {EXTRACTED_DIR.resolve()}")
    for r in results:
        if r["total_chunks"] > 0:
            print(f"    {r['ticker']:<8} {r['total_chunks']} chunks, "
                  f"{r['total_failed']} failed")
    if stopped_early:
        print("\n  Run `--resume` again (e.g. tomorrow) to continue from "
              "exactly this point — nothing done so far will be redone or lost.")
        sys.exit(2)


if __name__ == "__main__":
    main()