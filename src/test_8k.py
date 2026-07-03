import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extraction.section_splitter_8k import split_8k, ITEM_8K_LABELS, ITEM_8K_ORDER

tickers = [
    "AMZN", "TSLA", "HD",   "MCD",  "NKE",
    "SBUX", "TGT",  "LOW",  "BKNG", "GM", "XOM", "CVX", "COP", "SLB", "EOG",
    "PXD", "MPC", "PSX", "VLO", "OXY", "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "LLY", "BMY", "AMGN", "GILD", "CVS", "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "ADBE", "CRM", "INTC", "CSCO", "IBM"
]

# ── aggregate trackers ──
summary = {
    "extracted"     : [],
    "exhibits_only" : [],
    "failed"        : [],
    "no_8k"         : [],
    "no_files"      : [],
}
item_frequency   = defaultdict(int)   # how often each item type appears across all 8-Ks
event_type_count = defaultdict(int)   # event label frequency
failed_tickers   = []

ALL_RESULTS = {}  # ticker -> list of per-file results (for aggregate stats)

for ticker in tickers:
    print("=" * 70)
    print(f"TICKER: {ticker}")
    print("=" * 70)

    filing_dir   = Path(f"data/raw/filings/{ticker}")
    all_files    = list(filing_dir.glob("*.json")) if filing_dir.exists() else []
    eightk_files = sorted([f for f in all_files if f.name.startswith("8K_")])

    print(f"  8-K files on disk: {len(eightk_files)}")

    if not all_files:
        print(f"  No filings at all — check ingestion")
        summary["no_files"].append(ticker)
        print()
        continue

    if not eightk_files:
        print(f"  No 8-K files found")
        summary["no_8k"].append(ticker)
        print()
        continue

    ticker_results  = []
    status_counts   = defaultdict(int)
    items_seen      = defaultdict(int)

    # Test ALL 8-Ks for this ticker (not just first)
    for f in eightk_files:
        raw    = json.load(open(f))
        result = split_8k(raw["raw_html"], ticker=ticker, filing_date=raw["filing_date"])
        conf   = result["confidence"]
        status = conf["overall_status"]

        status_counts[status] += 1
        for iid in conf.get("items_found", []):
            items_seen[iid] += 1
            item_frequency[iid] += 1
        for label in conf.get("event_types", {}).values():
            event_type_count[label] += 1

        ticker_results.append(result)

    ALL_RESULTS[ticker] = ticker_results

    # Dominant status for this ticker
    dominant = max(status_counts, key=status_counts.get)
    summary[dominant].append(ticker)

    print(f"  Status breakdown: {dict(status_counts)}")
    print(f"  Items seen across all 8-Ks: {dict(sorted(items_seen.items()))}")

    # Show 3 most recent 8-Ks in detail
    print(f"\n  Showing last 3 8-Ks in detail:")
    for result in ticker_results[-3:]:
        conf  = result["confidence"]
        print(f"\n    [{result['filing_date']}] STATUS: {conf['overall_status']}")
        print(f"      Items found: {conf.get('items_found', [])}")
        for iid, label in conf.get("event_types", {}).items():
            text    = result["sections"].get(iid, "")
            preview = text[:120].replace("\n", " ")
            print(f"      [{iid}] {label}")
            print(f"             {len(text.split())} words — {preview}")

    print()

# ── Aggregate summary ──
print("#" * 70)
print("# AGGREGATE SUMMARY — 8-K FILINGS")
print("#" * 70)

total = sum(len(v) for k, v in summary.items() if k not in ("no_8k", "no_files"))
print(f"\n8-K tickers tested: {total}")
print(f"  extracted:      {len(summary['extracted'])}  {summary['extracted']}")
print(f"  exhibits_only:  {len(summary['exhibits_only'])}  {summary['exhibits_only']}")
print(f"  failed:         {len(summary['failed'])}  {summary['failed']}")
print(f"  no 8-K files:   {len(summary['no_8k'])}  {summary['no_8k']}")
print(f"  no files at all:{len(summary['no_files'])}  {summary['no_files']}")

print(f"\nMost common 8-K item types across all filings:")
sorted_items = sorted(item_frequency.items(), key=lambda x: -x[1])
for iid, count in sorted_items:
    label = ITEM_8K_LABELS.get(iid, f"Item {iid}")
    print(f"  Item {iid:<5} {count:>4}x  —  {label}")

print(f"\nMost common event types:")
sorted_events = sorted(event_type_count.items(), key=lambda x: -x[1])
for label, count in sorted_events[:15]:
    print(f"  {count:>4}x  {label}")