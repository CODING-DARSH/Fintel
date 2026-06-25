# import sys
# import json
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
# from src.extraction.section_splitter import split_filing
# tickers = [
#           "AMZN", "TSLA", "HD",   "MCD",  "NKE",
#         "SBUX", "TGT",  "LOW",  "BKNG", "GM","XOM", "CVX", "COP", "SLB", "EOG",
#         "PXD", "MPC", "PSX", "VLO", "OXY",       "JNJ", "PFE", "UNH", "ABBV", "MRK",
#         "LLY", "BMY", "AMGN","GILD", "CVS",        "AAPL", "MSFT", "GOOGL", "NVDA", "META",
#         "ADBE", "CRM",  "INTC",  "CSCO", "IBM"
# ]

# for ticker in tickers:
#     print("=" * 70)
#     print(f"TICKER: {ticker}")
#     print("=" * 70)

#     filing_dir = Path(f"data/raw/filings/{ticker}")
#     all_files  = list(filing_dir.glob("*.json")) if filing_dir.exists() else []
#     tenk_files = [f for f in all_files if f.name.startswith("10K_")]
#     eightk_files = [f for f in all_files if f.name.startswith("8K_")]

#     print(f"  Total files on disk: {len(all_files)}")
#     print(f"  10-K files: {len(tenk_files)}  |  8-K files: {len(eightk_files)}")

#     if not all_files:
#         print(f"  No filings found for {ticker} at all — check ingestion")
#         print()
#         continue

#     # ── Test 10-K if available ──
#     if tenk_files:
#         raw = json.load(open(tenk_files[0]))
#         result = split_filing(raw["raw_html"], ticker=ticker, filing_date=raw["filing_date"])
#         conf = result["confidence"]
#         print(conf)
#         print(f"\n  [10-K] {raw['filing_date']}  STATUS: {conf['overall_status']}")
#         print(f"    Order correct: {conf['order_correct']}  |  Length failures: {conf['length_failures']}")
#         if "1A" in result["sections"]:
#             t = result["sections"]["1A"]
#             print(f"    Item 1A: {len(t.split())} words — {t[:150]}")
#     else:
#         print(f"\n  [10-K] none found")

#     # ── Test 8-K if available ──
#     if eightk_files:
#         raw = json.load(open(eightk_files[0]))
#         result = split_filing(raw["raw_html"], ticker=ticker, filing_date=raw["filing_date"])
#         conf = result["confidence"]
#         print(f"\n  [8-K] {raw['filing_date']}  STATUS: {conf['overall_status']}")
#         print(f"    Sections found: {sorted(result['sections'].keys())}")
#         if "zero_occurrence" in conf:
#             print(f"    Zero occurrence: {conf['zero_occurrence']}")
#         if "missing_items" in conf:
#             print(f"    Missing items: {conf['missing_items']}")
#         # 8-Ks rarely have Item 1A/7/8 — they have their own item numbering
#         # (Item 1.01, 2.02, 5.02, 9.01 etc.) which our 10-K patterns won't match.
#         # An empty sections dict here is EXPECTED, not a bug — flagging it.
#         if not result["sections"]:
#             print(f"    NOTE: 8-Ks use different Item numbering (1.01, 2.02, 5.02, 9.01)")
#             print(f"          our patterns are 10-K specific — this is expected, not a failure")
#     else:
#         print(f"\n  [8-K] none found")

#     print()


import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extraction.section_splitter import split_filing, ITEM_ORDER

tickers = [
    "AMZN", "TSLA", "HD",   "MCD",  "NKE",
    "SBUX", "TGT",  "LOW",  "BKNG", "GM", "XOM", "CVX", "COP", "SLB", "EOG",
    "PXD", "MPC", "PSX", "VLO", "OXY",       "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "LLY", "BMY", "AMGN", "GILD", "CVS",        "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "ADBE", "CRM",  "INTC",  "CSCO", "IBM"
]

# ── aggregate trackers, filled in as we go ──
summary = {
    "verified": [],
    "needs_review": [],
    "failed": [],
    "incorporated_by_reference": [],
    "no_10k": [],
    "no_files": [],
}
order_failures = []          # tickers where found_order != expected subsequence
zero_occ_failures = defaultdict(list)   # item_id -> [tickers with zero occurrences]
length_failures_by_item = defaultdict(list)  # item_id -> [(ticker, reason)]


def diagnose_item(item_id, conf, sections):
    """
    Turn the raw confidence dict into a one-line, human-readable verdict
    for a single item, so failures are legible without re-deriving them
    by hand from occurrence_counts / length_check / found_order.
    """
    occ = conf["occurrence_counts"].get(item_id, 0)
    if occ == 0:
        return "MISSING — header never matched (pattern gap or structural miss)"

    if item_id not in sections:
        return f"NOT SELECTED — {occ} candidate(s) found but none survived TOC/cross-ref filtering"

    length_verdict = conf["length_check"].get(item_id, "no_bounds_defined")
    order_ok = item_id in conf["found_order"] and conf["order_correct"]

    flags = []
    if length_verdict not in ("ok", "no_bounds_defined"):
        flags.append(length_verdict)
    if item_id in conf["found_order"] and not conf["order_correct"]:
        flags.append("out of expected order — possible TOC/boundary mix-up")

    if not flags:
        return f"ok ({occ} candidate(s) seen, length ok)"
    return f"FLAGGED — {'; '.join(flags)} (saw {occ} candidate(s))"


for ticker in tickers:
    print("=" * 70)
    print(f"TICKER: {ticker}")
    print("=" * 70)

    filing_dir = Path(f"data/raw/filings/{ticker}")
    all_files = list(filing_dir.glob("*.json")) if filing_dir.exists() else []
    tenk_files = [f for f in all_files if f.name.startswith("10K_")]
    eightk_files = [f for f in all_files if f.name.startswith("8K_")]

    print(f"  Total files on disk: {len(all_files)}")
    print(f"  10-K files: {len(tenk_files)}  |  8-K files: {len(eightk_files)}")

    if not all_files:
        print(f"  No filings found for {ticker} at all — check ingestion")
        summary["no_files"].append(ticker)
        print()
        continue

    # ── Test 10-K if available ──
    if tenk_files:
        raw = json.load(open(tenk_files[0]))
        result = split_filing(raw["raw_html"], ticker=ticker, filing_date=raw["filing_date"])
        conf = result["confidence"]
        sections = result["sections"]

        status = conf["overall_status"]
        summary[status].append(ticker)

        if status == "incorporated_by_reference":
            print(f"\n  [10-K] {raw['filing_date']}  STATUS: incorporated_by_reference")
            print(f"    Content lives in a separate exhibit — not extractable from this HTML.")
            print(f"    Earliest match at {conf.get('earliest_match_pct', '?')}% of document.")
        else:
            if not conf["order_correct"]:
                order_failures.append(ticker)
            for item_id in conf["zero_occurrence"]:
                zero_occ_failures[item_id].append(ticker)
            for item_id in conf["length_failures"]:
                length_failures_by_item[item_id].append((ticker, conf["length_check"][item_id]))

            print(f"\n  [10-K] {raw['filing_date']}  STATUS: {status}")
            print(f"    Order correct: {conf['order_correct']}")
            print(f"    Expected order subsequence: {[i for i in ITEM_ORDER if i in conf['found_order']]}")
            print(f"    Found order:                {conf['found_order']}")

            print(f"    Per-item diagnosis:")
            for item_id in ITEM_ORDER:
                verdict = diagnose_item(item_id, conf, sections)
                print(f"      Item {item_id:<3}: {verdict}")

            print(f"    Section previews:")
            for item_id in ITEM_ORDER:
                if item_id in sections:
                    t = sections[item_id]
                    preview = t[:120].replace("\n", " ")
                    print(f"      [{item_id:<3}] {len(t.split()):>6} words — {preview}")
    else:
        print(f"\n  [10-K] none found")
        summary["no_10k"].append(ticker)

    # ── Test 8-K if available ──
    if eightk_files:
        raw = json.load(open(eightk_files[0]))
        result = split_filing(raw["raw_html"], ticker=ticker, filing_date=raw["filing_date"])
        conf = result["confidence"]
        print(f"\n  [8-K] {raw['filing_date']}  STATUS: {conf['overall_status']}")
        print(f"    Sections found: {sorted(result['sections'].keys())}")
        if not result["sections"]:
            print(f"    NOTE: 8-Ks use different Item numbering (1.01, 2.02, 5.02, 9.01)")
            print(f"          our patterns are 10-K specific — this is expected, not a failure")
    else:
        print(f"\n  [8-K] none found")

    print()

# ── Aggregate diagnostics across all tickers ──
print("#" * 70)
print("# AGGREGATE SUMMARY")
print("#" * 70)

total_with_10k = len(summary["verified"]) + len(summary["needs_review"]) + len(summary["failed"]) + len(summary["incorporated_by_reference"])
print(f"\n10-Ks tested: {total_with_10k}")
print(f"  verified:                  {len(summary['verified'])}  {summary['verified']}")
print(f"  needs_review:              {len(summary['needs_review'])}  {summary['needs_review']}")
print(f"  failed:                    {len(summary['failed'])}  {summary['failed']}")
print(f"  incorporated_by_reference: {len(summary['incorporated_by_reference'])}  {summary['incorporated_by_reference']}")
print(f"  no 10-K file:              {len(summary['no_10k'])}  {summary['no_10k']}")
print(f"  no files at all:           {len(summary['no_files'])}  {summary['no_files']}")

print(f"\nOrder-correct failures ({len(order_failures)} tickers): {order_failures}")

print(f"\nZero-occurrence (header never matched) by item:")
for item_id in ITEM_ORDER:
    if item_id in zero_occ_failures:
        print(f"  Item {item_id:<3}: {len(zero_occ_failures[item_id])} tickers — {zero_occ_failures[item_id]}")

print(f"\nLength-bound failures by item:")
for item_id in ITEM_ORDER:
    if item_id in length_failures_by_item:
        print(f"  Item {item_id}:")
        for ticker, reason in length_failures_by_item[item_id]:
            print(f"    {ticker:<6} — {reason}")