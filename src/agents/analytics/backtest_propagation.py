# =============================================================================
# src/agents/analytics/backtest_propagation.py
# =============================================================================
# Validates graph_propagation.py's rule-based edge weights against REAL,
# dated, historical disruption events — using price data you already have
# access to via yfinance (same library market_connector.py uses).
#
# This does NOT try to validate exact confidence VALUES (0.42 is not being
# checked against "the true number") — that would be a much stronger and
# less honest claim than the data supports. Instead it checks the weaker,
# more defensible claim: does the RELATIVE RANKING propagation produces
# correlate with which companies actually moved more in the market
# following the event. That's a rank correlation (Spearman), not a
# point-value calibration.
#
# KNOWN LIMITATION, stated plainly rather than hidden: the graph is a
# current/static snapshot, not a point-in-time historical graph. Testing
# a 2021 event against today's graph assumes relationships that exist
# now (e.g. a supplier link) also existed then. Reasonable for
# long-standing supplier relationships, noisier for anything that
# changed since — this is a real caveat on the results, not a bug to fix.
#
# Usage:
#   python src/agents/analytics/backtest_propagation.py
#   python src/agents/analytics/backtest_propagation.py --dry-run   (no
#       network/DB calls — exercises the statistics/report logic only,
#       using fake data, for structural verification)
# =============================================================================

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

# Running this file directly (python src/agents/analytics/backtest_propagation.py)
# only adds THIS file's own directory to sys.path, not the project root —
# unlike root-level scripts (e.g. run_agent_query.py) whose own directory
# IS the project root. Since this file lives 3 levels under src/, walk up
# to the project root explicitly so `from src...` imports resolve
# regardless of how/where this is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known, dated, verifiable disruption events — hand-picked because they
# actually happened, are well-documented, and have an obvious geography/
# input anchor to test propagation against. Expand this list over time
# as more good test cases come to mind; four is a starting point, not a
# ceiling.
# ---------------------------------------------------------------------------

@dataclass
class BacktestEvent:
    name: str
    event_date: str          # YYYY-MM-DD, the day the disruption became public
    start_label: str          # "Geography" | "Input" | "Event"
    start_id: str             # must match your graph's id normalization
    description: str
    window_days: int = 10      # trading days after event to measure reaction
    control_ticker: str = "SPY"  # baseline to compute abnormal (excess) return


KNOWN_EVENTS = [
    BacktestEvent(
        name="2021 Suez Canal blockage",
        event_date="2021-03-23",
        start_label="Geography",
        start_id="egypt",
        description="Ever Given ran aground, blocking the Suez Canal for ~6 days, "
                     "disrupting global shipping/logistics.",
        window_days=15,
    ),
    BacktestEvent(
        name="2022 Shanghai COVID lockdown",
        event_date="2022-03-28",
        start_label="Geography",
        start_id="china",
        description="Extended Shanghai lockdown disrupted manufacturing and exports "
                     "from one of China's largest industrial/port hubs.",
        window_days=20,
    ),
    BacktestEvent(
        name="2018 US steel & aluminum tariffs",
        event_date="2018-03-08",
        start_label="Input",
        start_id="steel",
        description="Section 232 tariffs on imported steel and aluminum, raising "
                     "input costs for steel-consuming manufacturers.",
        window_days=15,
    ),
    BacktestEvent(
        name="2021 global chip shortage escalation",
        event_date="2021-02-08",
        start_label="Input",
        start_id="semiconductors",
        description="Automakers/electronics manufacturers began publicly cutting "
                     "production guidance due to chip shortages.",
        window_days=20,
    ),
    BacktestEvent(
        name="2022 McDonald's Russia market exit",
        event_date="2022-05-16",
        start_label="Geography",
        start_id="russia",
        description="McDonald's announced it would sell its Russian business "
                     "entirely, exiting the market after the invasion of Ukraine. "
                     "Confirmed real graph anchor — MCD's own filing directly "
                     "discusses this exit as a geopolitical_risk_market_exit "
                     "risk factor, found via discover_connections.py rather "
                     "than guessed.",
        window_days=15,
    ),
]


# ---------------------------------------------------------------------------
# Price reaction fetching — reuses the same yfinance access pattern as
# market_connector.py, just windowed around a specific historical date
# instead of "most recent."
# ---------------------------------------------------------------------------

def fetch_return(ticker: str, event_date: str, window_days: int) -> Optional[float]:
    """
    Percent price return from event_date's close to window_days trading
    days later. Returns None if data isn't available (delisted, too
    recent, ticker typo, rate-limited, etc.) — never fabricates a number.
    """
    try:
        import yfinance as yf
        start = datetime.strptime(event_date, "%Y-%m-%d")
        end   = start + timedelta(days=window_days * 2)  # buffer for weekends/holidays
        hist  = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                           end=end.strftime("%Y-%m-%d"))
        if hist.empty or len(hist) < 2:
            return None

        base_price = float(hist.iloc[0]["Close"])
        idx = min(window_days, len(hist) - 1)
        end_price = float(hist.iloc[idx]["Close"])

        return ((end_price - base_price) / base_price) * 100
    except Exception as e:
        log.warning(f"fetch_return failed for {ticker}: {e}")
        return None
    finally:
        # Yahoo's endpoints rate-limit aggressively against rapid-fire
        # requests (confirmed via a real run: even SPY, the most liquid
        # ticker in existence, came back 429'd after ~70 requests with
        # no delay between them). This delay is deliberate, not
        # incidental — same discipline as extractor.py's REQUEST_DELAY.
        time.sleep(1.0)


def compute_abnormal_return(ticker: str, event: BacktestEvent,
                             control_return: Optional[float]) -> Optional[float]:
    """Company's raw return minus the control ticker's return over the
    same window — isolates the event-specific reaction from broad
    market movement in that period. control_return is fetched ONCE per
    event by the caller and passed in, rather than re-fetched per
    company — SPY's return for a given event+window doesn't change
    based on which company you're comparing it to, so re-fetching it
    per company was pure waste that also made rate-limiting worse."""
    company_return = fetch_return(ticker, event.event_date, event.window_days)
    if company_return is None or control_return is None:
        return None
    return company_return - control_return


# ---------------------------------------------------------------------------
# Rank correlation — implemented manually (no scipy dependency assumed)
# since this needs to run inside the pipeline container without knowing
# in advance whether scipy is installed there.
# ---------------------------------------------------------------------------

def spearman_correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    """
    Manual Spearman rank correlation. Returns None if fewer than 3 pairs
    (not enough data for a meaningful correlation) or if either series
    has zero variance (e.g. all propagation scores identical).
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None

    def rank(values: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)

    if len(set(rx)) == 1 or len(set(ry)) == 1:
        return None

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = sum((r - mean_rx) ** 2 for r in rx) ** 0.5
    std_y = sum((r - mean_ry) ** 2 for r in ry) ** 0.5
    if std_x == 0 or std_y == 0:
        return None

    return cov / (std_x * std_y)


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

@dataclass
class EventResult:
    event: BacktestEvent
    pairs: list[tuple] = field(default_factory=list)  # (ticker, score, abnormal_return)
    correlation: Optional[float] = None


def run_backtest(events: list[BacktestEvent]) -> list[EventResult]:
    from src.retrieval.graph_retriever import GraphRetriever
    from src.agents.analytics.graph_propagation import find_paths

    graph = GraphRetriever()
    results = []

    for event in events:
        log.info(f"--- {event.name} ---")

        if not graph.node_exists(event.start_label, event.start_id):
            log.warning(
                f"  Start entity {event.start_label}:{event.start_id} not found "
                f"in graph — skipping (this is informative, not an error: it "
                f"means this event isn't represented in your extracted data yet)"
            )
            results.append(EventResult(event=event))
            continue

        prop_result = find_paths(graph, event.start_label, event.start_id, max_depth=4)
        if not prop_result.companies_reached:
            log.info(f"  No companies reached from this start entity — skipping")
            results.append(EventResult(event=event))
            continue

        pairs = []
        control_return = fetch_return(event.control_ticker, event.event_date, event.window_days)
        if control_return is None:
            log.warning(
                f"  Control ticker {event.control_ticker} unavailable for this "
                f"event/window — cannot compute abnormal returns, skipping"
            )
            results.append(EventResult(event=event))
            continue

        for ticker in prop_result.companies_reached:
            score = prop_result.paths[ticker][0].path_score
            abnormal = compute_abnormal_return(ticker, event, control_return)
            if abnormal is None:
                continue
            pairs.append((ticker, score, abnormal))
            log.info(f"  {ticker}: propagation_score={score:.3f}  abnormal_return={abnormal:+.2f}%")

        corr = None
        if len(pairs) >= 3:
            scores = [p[1] for p in pairs]
            abs_returns = [abs(p[2]) for p in pairs]
            corr = spearman_correlation(scores, abs_returns)

        results.append(EventResult(event=event, pairs=pairs, correlation=corr))

    graph.close()
    return results


def print_report(results: list[EventResult]):
    print("\n" + "=" * 70)
    print("PROPAGATION BACKTEST REPORT")
    print("=" * 70)
    print(
        "Checks whether propagation's score RANKING correlates with actual\n"
        "price reaction magnitude — not whether exact scores are 'correct'.\n"
        "Positive correlation = higher-scored companies moved more, on\n"
        "average, which is the honest bar for 'is rule-based weighting\n"
        "directionally useful.'\n"
    )

    all_scores, all_abs_returns = [], []

    for r in results:
        print(f"\n{r.event.name} ({r.event.event_date})")
        print(f"  {r.event.description}")
        if not r.pairs:
            print("  No usable data for this event (see log above for why).")
            continue
        for ticker, score, abnormal in sorted(r.pairs, key=lambda p: -p[1]):
            print(f"    {ticker:<8} score={score:.3f}   abnormal_return={abnormal:+7.2f}%")
        if r.correlation is not None:
            print(f"  Spearman correlation (score vs |abnormal return|): {r.correlation:+.3f}")
        else:
            print(f"  Not enough data points for a correlation (need 3+, got {len(r.pairs)})")
        all_scores.extend(p[1] for p in r.pairs)
        all_abs_returns.extend(abs(p[2]) for p in r.pairs)

    print("\n" + "-" * 70)
    if len(all_scores) >= 3:
        pooled = spearman_correlation(all_scores, all_abs_returns)
        if pooled is not None:
            print(f"POOLED correlation across all events: {pooled:+.3f}")
            if pooled > 0.3:
                print("  -> Meaningful positive signal: current weights are directionally useful.")
            elif pooled > 0:
                print("  -> Weak positive signal: directionally okay, worth more events to confirm.")
            else:
                print("  -> No positive signal: current rule-based weights are not tracking real "
                      "outcomes — worth revisiting the weight table, or this may need more/better "
                      "test events before concluding anything.")
        else:
            print("  Pooled correlation undefined (insufficient variance).")
    else:
        print(f"Not enough total data points ({len(all_scores)}) for a pooled correlation.")
    print("=" * 70)


def _dry_run():
    """Structural smoke test with fake data — no network or DB needed.
    Verifies the statistics/report logic works correctly on its own."""
    fake_event = BacktestEvent(
        name="FAKE — dry run test", event_date="2021-01-01",
        start_label="Geography", start_id="testland",
        description="synthetic test event",
    )
    fake_result = EventResult(
        event=fake_event,
        pairs=[("AAA", 0.9, 8.5), ("BBB", 0.6, 3.2), ("CCC", 0.3, 1.1), ("DDD", 0.1, -0.5)],
    )
    scores = [p[1] for p in fake_result.pairs]
    abs_returns = [abs(p[2]) for p in fake_result.pairs]
    fake_result.correlation = spearman_correlation(scores, abs_returns)
    print_report([fake_result])


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        _dry_run()
    else:
        results = run_backtest(KNOWN_EVENTS)
        print_report(results)