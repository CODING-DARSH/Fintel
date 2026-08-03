# =============================================================================
# src/agents/analytics/discover_connections.py
# =============================================================================
# Before adding more backtest events, find out what NVDA, AAPL, TSLA, HD,
# NKE, MCD are ACTUALLY connected to in the real graph — rather than
# guessing more Geography/Input node names and repeating the earlier
# "egypt"/"steel" miss (both plausible-sounding names that didn't exist).
#
# For each target ticker, lists every DEPENDS_ON / EXPOSED_TO /
# OPERATES_IN / CONCENTRATED_IN / SOURCED_FROM neighbor — the real
# Input/Geography/Risk nodes connected to it — so backtest events can be
# anchored on confirmed-real node ids instead of assumptions.
#
# Usage:
#   python src/agents/analytics/discover_connections.py
# =============================================================================

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)  # quiet — this is a report tool
log = logging.getLogger(__name__)

TARGET_TICKERS = ["NVDA", "AAPL", "TSLA", "HD", "NKE", "MCD"]

REL_TYPES = ["DEPENDS_ON", "EXPOSED_TO", "OPERATES_IN", "CONCENTRATED_IN",
             "HAS_CONCENTRATION", "SUPPLIES_TO", "BUYS_FROM"]


def discover(graph, ticker: str) -> dict:
    """Returns {rel_type: [(neighbor_label, neighbor_id, neighbor_name), ...]}"""
    results = {}
    rel_pattern = "|".join(REL_TYPES)
    rows = graph._run(
        f"""
        MATCH (c:Company {{ticker: $ticker}})-[r:{rel_pattern}]-(n)
        RETURN type(r) AS rel_type, labels(n) AS labels,
               coalesce(n.input_id, n.geo_id, n.risk_id, n.market_id, n.ticker) AS id,
               coalesce(n.description, n.name, n.ticker) AS name
        LIMIT 100
        """,
        {"ticker": ticker},
    )
    for row in rows:
        rel = row["rel_type"]
        results.setdefault(rel, [])
        entry = (row["labels"][0] if row["labels"] else "?", row["id"], row["name"])
        if entry not in results[rel]:
            results[rel].append(entry)
    return results


def main():
    from src.retrieval.graph_retriever import GraphRetriever

    graph = GraphRetriever()

    print("=" * 70)
    print("ENTITY CONNECTION DISCOVERY — for building real backtest anchors")
    print("=" * 70)

    all_inputs = set()
    all_geos = set()

    for ticker in TARGET_TICKERS:
        print(f"\n--- {ticker} ---")
        conns = discover(graph, ticker)
        if not conns:
            print("  (no DEPENDS_ON/EXPOSED_TO/OPERATES_IN/etc. relationships found)")
            continue
        for rel_type, neighbors in conns.items():
            print(f"  {rel_type}:")
            for label, nid, name in neighbors[:15]:
                print(f"    [{label}] {nid}  ({name})")
                if label == "Input":
                    all_inputs.add(nid)
                elif label == "Geography":
                    all_geos.add(nid)

    print("\n" + "=" * 70)
    print("SUMMARY — confirmed-real Input/Geography ids across all 6 tickers")
    print("=" * 70)
    print(f"\nInputs ({len(all_inputs)}):")
    for i in sorted(all_inputs):
        print(f"  - {i}")
    print(f"\nGeographies ({len(all_geos)}):")
    for g in sorted(all_geos):
        print(f"  - {g}")

    graph.close()


if __name__ == "__main__":
    main()