# # =============================================================================
# # src/extraction/section_splitter.py
# # =============================================================================
# # Splits a raw SEC filing into named Item sections.
# #
# # APPROACH HISTORY (documented because each failure was genuinely instructive):
# #   1. "last text occurrence"     — broke: cross-refs cluster late in Item 7
# #   2. "second text occurrence"   — broke: early cross-ref before real header
# #   3. "phrase filtering"         — broke: unbounded list of cross-ref phrasings
# #   4. "position windows (%)"     — broke: cross-ref + header in same window
# #   5. "sequential ordering"      — broke: multiple candidates between two
# #                                    correctly-anchored neighbors, still guessing
# #   6. "isolated DOM elements"    — broke into 4 generic failure classes:
# #         a) TOC capture       — TOC rows are short isolated elements too
# #         b) Cross-ref capture — hyperlinked cross-refs are short+isolated too
# #         c) Boundary failure  — position offsets computed by double-counting
# #                                  nested-tag text (get_document_position summed
# #                                  every tag's get_text(), counting child text
# #                                  twice: once for the child, once for the
# #                                  parent wrapping it)
# #         d) Header miss       — tag-level get_text() scanning misses headers
# #                                  that are plain text nodes sitting inside a
# #                                  large parent tag (parent text too long to
# #                                  pass the isolation filter, but the specific
# #                                  text NODE itself is short and isolated)
# #
# # CURRENT APPROACH (v7): work at the TEXT-NODE level, not the tag level.
# #   - Position truth comes from a single ordered list of non-empty text
# #     nodes (NavigableStrings), built ONCE via soup.find_all(string=True).
# #     This is inherently document-order and never double-counts, because
# #     each piece of text exists as exactly one node regardless of how many
# #     ancestor tags wrap it. This directly fixes the boundary-failure bug.
# #   - Header candidates are detected at BOTH the text-node level (catches
# #     headers with no dedicated wrapper tag — fixes header-detection-miss)
# #     AND the tag level (catches headers split across sibling inline tags,
# #     e.g. <b>ITEM 1A.</b><span>RISK FACTORS</span>).
# #   - Cross-reference hyperlinks are explicitly excluded: any candidate
# #     that is an <a> tag, or has an <a> as its immediate text-bearing
# #     ancestor, is dropped — this is the single highest-value fix, since
# #     SEC filings hyperlink nearly every cross-reference.
# #   - TOC rows are filtered out using three independent signals instead of
# #     one fragile "prefer last" guess: (1) inside a <table>, (2) text ends
# #     in a trailing page-number digit, (3) falls in the first 8% of the
# #     document's text nodes (TOC is always near the top). A candidate only
# #     needs to trip ONE of these to be down-ranked as TOC.
# #   - Among whatever survives, we pick the FIRST remaining candidate in
# #     document order — not the last. Once TOC + cross-refs are correctly
# #     excluded, there should be exactly one real header left, and "first"
# #     is the structurally correct choice (the header that actually starts
# #     the section), whereas "last" was a workaround for the old method's
# #     inability to exclude TOC/cross-ref candidates cleanly.

# import re
# import json
# import logging
# import warnings
# from pathlib import Path
# from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

# warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

# ITEM_PATTERNS = {
#     "1"  : r"item\s*1\.?\s*business",
#     "1A" : r"item\s*1a\.?\s*risk\s*factors",
#     "1B" : r"item\s*1b\.?\s*unresolved\s*staff\s*comments",
#     "2"  : r"item\s*2\.?\s*properties",
#     "3"  : r"item\s*3\.?\s*legal\s*proceedings",
#     "5"  : r"item\s*5\.?\s*market\s*for\s*registrant",
#     "7"  : r"item\s*7\.?\s*management.s\s*disc",
#     "7A" : r"item\s*7a\.?\s*quantitative\s*and\s*qualitative",
#     "8"  : r"item\s*8\.?\s*financial\s*statements",
#     "9"  : r"item\s*9\.?\s*changes\s*in\s*and\s*disagreements",
#     "9A" : r"item\s*9a\.?\s*controls\s*and\s*procedures",
#     "10" : r"item\s*10\.?\s*directors",
# }

# ITEM_ORDER = ["1", "1A", "1B", "2", "3", "5", "7", "7A", "8", "9", "9A", "10"]

# LENGTH_BOUNDS = {
#     "1"  : (200, 15000),
#     "1A" : (500, 25000),
#     "1B" : (0,   2000),
#     "2"  : (50,  3000),
#     "3"  : (50,  8000),
#     "5"  : (50,  5000),
#     "7"  : (500, 30000),
#     "7A" : (50,  6000),
#     "8"  : (50,  80000),
#     "9"  : (0,   2000),
#     "9A" : (50,  3000),
# }

# # Max length (chars) for a node/element's OWN text to count as an
# # "isolated header" candidate. ~26 chars for a real header; 200 gives
# # generous margin while still excluding paragraph-length cross-refs.
# MAX_HEADER_ELEMENT_LENGTH = 200

# # TOC is always near the top of the document. If a candidate falls in
# # this fraction of the document's text nodes, it's TOC-suspect.
# TOC_REGION_FRACTION = 0.08

# # A trailing run of digits (optionally preceded by punctuation/space) on
# # an otherwise-matching short string is almost always a TOC page number,
# # e.g. "Item 1A. Risk Factors 23" or "Item 1A.Risk Factors2".
# TRAILING_PAGE_NUMBER = re.compile(r"\d+\s*$")


# def _is_cross_reference(node) -> bool:
#     """
#     True if this text node / tag sits inside a hyperlink. SEC filings
#     hyperlink nearly every cross-reference ("see Item 1A. Risk Factors"),
#     and the anchor text alone is short + isolated, so without this check
#     it sails through the length filter looking exactly like a real header.
#     """
#     parent = node.parent if isinstance(node, NavigableString) else node
#     return parent is not None and parent.find_parent("a") is not None or (
#         parent is not None and parent.name == "a"
#     )


# def _is_in_table(node) -> bool:
#     parent = node.parent if isinstance(node, NavigableString) else node
#     return parent is not None and parent.find_parent("table") is not None


# def build_text_node_index(soup: BeautifulSoup):
#     """
#     Single ordered list of non-empty text nodes. This is the ONE source
#     of truth for document position — no double counting, because each
#     string exists exactly once here regardless of tag nesting.

#     Returns:
#         nodes: list of NavigableString objects, in document order
#         node_id_to_idx: dict mapping id(node) -> index in `nodes`
#     """
#     nodes = [n for n in soup.find_all(string=True) if n.strip()]
#     node_id_to_idx = {id(n): i for i, n in enumerate(nodes)}
#     return nodes, node_id_to_idx


# def _first_text_node_of(tag, node_id_to_idx):
#     """Index (in the global node list) of a tag's first non-empty text descendant."""
#     for s in tag.find_all(string=True):
#         if s.strip() and id(s) in node_id_to_idx:
#             return node_id_to_idx[id(s)]
#     return None


# def find_header_candidates(soup: BeautifulSoup, nodes, node_id_to_idx) -> dict:
#     """
#     Collect header candidates from BOTH text nodes and tags:
#       - text-node level catches headers with no dedicated wrapper tag
#         (fixes header-detection-miss, e.g. Item 5 going to zero)
#       - tag level catches headers split across sibling inline tags

#     Each candidate carries enough metadata to classify it as
#     TOC / cross-ref / real further down the pipeline.
#     """
#     candidates_by_item = {item_id: [] for item_id in ITEM_PATTERNS}
#     total_nodes = len(nodes)
#     seen_positions = set()  # (item_id, node_idx) dedup key

#     def consider(text_raw, node_idx, source_obj, tag_name):
#         text = re.sub(r"\s+", " ", text_raw).strip()
#         if not text or len(text) > MAX_HEADER_ELEMENT_LENGTH:
#             return
#         for item_id, pattern in ITEM_PATTERNS.items():
#             if re.search(pattern, text, re.IGNORECASE):
#                 key = (item_id, node_idx)
#                 if key in seen_positions:
#                     continue
#                 seen_positions.add(key)
#                 candidates_by_item[item_id].append({
#                     "text": text,
#                     "node_idx": node_idx,
#                     "tag_name": tag_name,
#                     "is_link": _is_cross_reference(source_obj),
#                     "in_table": _is_in_table(source_obj),
#                     "is_toc_region": (node_idx / total_nodes) < TOC_REGION_FRACTION
#                                       if total_nodes else False,
#                     "has_trailing_digits": bool(TRAILING_PAGE_NUMBER.search(text)),
#                 })

#     # 1) Text-node level (catches headers with no isolating wrapper tag)
#     for s in nodes:
#         idx = node_id_to_idx[id(s)]
#         consider(str(s), idx, s, "#text")

#     # 2) Tag level (catches headers split across sibling inline tags)
#     for tag in soup.find_all(True):
#         text = tag.get_text(strip=True)
#         if not text or len(text) > MAX_HEADER_ELEMENT_LENGTH:
#             continue
#         idx = _first_text_node_of(tag, node_id_to_idx)
#         if idx is None:
#             continue
#         consider(text, idx, tag, tag.name)

#     return candidates_by_item


# def select_real_header(item_id: str, candidates: list) -> dict | None:
#     """
#     Among candidates for one item, exclude cross-refs and TOC entries,
#     then take the FIRST remaining candidate in document order — that's
#     the header that actually starts the section.
#     """
#     if not candidates:
#         return None

#     candidates = sorted(candidates, key=lambda c: c["node_idx"])

#     # Hard exclude: hyperlinked cross-references are never real headers.
#     non_link = [c for c in candidates if not c["is_link"]]
#     if not non_link:
#         # Everything was a hyperlink (unusual) — fall back to first anyway
#         return candidates[0]

#     # Score TOC-suspicion; prefer candidates that trip ZERO TOC signals.
#     def toc_score(c):
#         return int(c["in_table"]) + int(c["has_trailing_digits"]) + int(c["is_toc_region"])

#     clean = [c for c in non_link if toc_score(c) == 0]
#     if clean:
#         return clean[0]

#     # Nothing fully clean — take the lowest TOC-suspicion candidate,
#     # preferring later document position as a tiebreaker (TOC is early).
#     non_link.sort(key=lambda c: (toc_score(c), -c["node_idx"]))
#     return non_link[0]


# def split_filing(raw_html: str, ticker: str = "", filing_date: str = "") -> dict:
#     """
#     Main entry point. Builds a single ordered text-node index (the source
#     of truth for position), finds header candidates at both text-node and
#     tag granularity, filters out cross-refs/TOC, then extracts section
#     text by slicing the node list between consecutive real headers.
#     """
#     soup = BeautifulSoup(raw_html, "lxml")
#     nodes, node_id_to_idx = build_text_node_index(soup)

#     candidates_by_item = find_header_candidates(soup, nodes, node_id_to_idx)
#     occurrence_counts = {k: len(v) for k, v in candidates_by_item.items()}

#     real_headers = {}
#     for item_id, candidates in candidates_by_item.items():
#         chosen = select_real_header(item_id, candidates)
#         if chosen:
#             real_headers[item_id] = chosen

#     # Position = node index directly. No re-searching flattened text,
#     # no char-offset math, no double counting.
#     positions = {item_id: info["node_idx"] for item_id, info in real_headers.items()}
#     sorted_items = sorted(positions.items(), key=lambda x: x[1])

#     node_texts = [str(n).strip() for n in nodes]

#     sections = {}
#     for i, (item_id, start_idx) in enumerate(sorted_items):
#         if item_id == "10":
#             continue  # boundary marker only
#         end_idx = sorted_items[i + 1][1] if i + 1 < len(sorted_items) else len(nodes)
#         section_text = " ".join(t for t in node_texts[start_idx:end_idx] if t)
#         section_text = re.sub(r"\s+", " ", section_text).strip()
#         sections[item_id] = section_text

#     # ── Verification ──
#     found_order = [item_id for item_id, _ in sorted_items if item_id != "10"]
#     expected_subsequence = [item for item in ITEM_ORDER if item in found_order]
#     order_correct = (found_order == expected_subsequence)

#     length_check = {}
#     for item_id, text in sections.items():
#         word_count = len(text.split())
#         bounds = LENGTH_BOUNDS.get(item_id)
#         if bounds is None:
#             length_check[item_id] = "no_bounds_defined"
#         elif word_count < bounds[0]:
#             length_check[item_id] = f"too_short ({word_count}w, expected>={bounds[0]})"
#         elif word_count > bounds[1]:
#             length_check[item_id] = f"too_long ({word_count}w, expected<={bounds[1]})"
#         else:
#             length_check[item_id] = "ok"

#     length_failures = [k for k, v in length_check.items() if v not in ("ok", "no_bounds_defined")]
#     zero_occurrence = [k for k, v in occurrence_counts.items() if v == 0]

#     overall_status = "verified"
#     if zero_occurrence:
#         overall_status = "needs_review"
#     if not order_correct:
#         overall_status = "needs_review"
#     if length_failures:
#         overall_status = "needs_review"
#     if not sections:
#         overall_status = "failed"

#     confidence = {
#         "occurrence_counts" : occurrence_counts,
#         "zero_occurrence"   : zero_occurrence,
#         "found_order"       : found_order,
#         "order_correct"     : order_correct,
#         "length_check"      : length_check,
#         "length_failures"   : length_failures,
#         "overall_status"    : overall_status,
#     }

#     return {
#         "ticker"      : ticker,
#         "filing_date" : filing_date,
#         "sections"    : sections,
#         "confidence"  : confidence,
#     }


# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#     test_path = Path("data/raw/filings/AAPL")
#     json_files = list(test_path.glob("10K_*.json"))
#     if not json_files:
#         print("No AAPL 10-K found")
#         sys.exit(1)

#     raw = json.load(open(json_files[0]))
#     result = split_filing(raw["raw_html"], ticker="AAPL", filing_date=raw["filing_date"])

#     print("="*70)
#     print("VERIFICATION RESULT")
#     print("="*70)
#     print(json.dumps(result["confidence"], indent=2))
#     print()
#     for item_id, text in result["sections"].items():
#         print(f"--- Item {item_id} ({len(text.split())} words) ---")
#         print(text[:200] + "...")
#         print()

# =============================================================================
# src/extraction/section_splitter.py
# =============================================================================
# Splits a raw SEC filing into named Item sections.
#
# APPROACH HISTORY (documented because each failure was genuinely instructive):
#   1-5. Various flattened-text heuristics — all broke because a
#        cross-reference and a real header look identical as plain text
#        but are structurally different in the DOM.
#   6. Isolated-DOM-element approach — fixed most cases, but a 40-ticker
#      test run exposed 4 GENERIC failure classes:
#         a) TOC capture       — TOC rows are short isolated elements too
#         b) Cross-ref capture — hyperlinked cross-refs are short+isolated too
#         c) Boundary failure  — position offsets computed by double-counting
#                                  nested-tag text
#         d) Header miss       — tag-level scanning misses headers that are
#                                  plain text nodes inside a large parent tag
#   7. Text-node-level position truth + <a> exclusion + TOC scoring — fixed
#      all 4 classes (0 "failed" results across 40 tickers), but exposed
#      3 FURTHER classes on the needs_review tickers:
#         e) Combined headers   — O&G filers merge "Item 1 and 2. Business
#                                   and Properties" into one header; neither
#                                   the Item 1 nor Item 2 pattern matches it
#         f) Phrasing variance  — e.g. Item 5's subtitle varies by filer
#                                   ("Market for the Registrant's..." vs
#                                   "Market Information" vs "Market for
#                                   Common Stock") and a single rigid regex
#                                   missed several of them
#         g) Stub-header trap   — some filers (MCD, LOW, INTC pattern: EVERY
#                                   item ends up too_short in lockstep) have
#                                   more than one short isolated occurrence
#                                   of a header's text, and the "first
#                                   non-TOC, non-link" candidate picked is a
#                                   stub sitting right next to the NEXT
#                                   item's header, not the one followed by
#                                   real narrative content.
#
# CURRENT APPROACH (v8):
#   - Position truth: one ordered list of non-empty text nodes
#     (NavigableStrings), built once. No double counting — fixes (c).
#   - Header candidates collected at both text-node and tag level — fixes (d).
#   - Hyperlinked candidates (<a> tags / descendants of <a>) hard-excluded —
#     fixes (b), the dominant cause of cross-ref capture in modern filings.
#   - TOC candidates scored on 3 independent signals (in <table>, trailing
#     page-number digits, falls in first 8% of document) rather than one
#     fragile "prefer last" guess — fixes (a).
#   - ITEM_PATTERNS now holds a LIST of alternate regexes per item, so
#     filer-specific phrasing variance and combined Item 1/2 headers are
#     handled by adding alternates, not per-ticker patches — fixes (e), (f).
#   - After initial section extraction, any item whose section comes out
#     implausibly short (below its LENGTH_BOUNDS minimum) automatically
#     retries with the NEXT surviving candidate for that item (by document
#     order) instead of giving up — fixes (g). This also incidentally
#     repairs several order_correct failures, since picking the wrong stub
#     occurrence was what threw off relative ordering in those filings too.

# =============================================================================
# src/extraction/section_splitter.py
# =============================================================================
# Splits a raw SEC filing into named Item sections.
#
# APPROACH HISTORY (documented because each failure was genuinely instructive):
#   1-5. Various flattened-text heuristics — all broke because a
#        cross-reference and a real header look identical as plain text
#        but are structurally different in the DOM.
#   6. Isolated-DOM-element approach — fixed most cases, but a 40-ticker
#      test run exposed 4 GENERIC failure classes:
#         a) TOC capture       — TOC rows are short isolated elements too
#         b) Cross-ref capture — hyperlinked cross-refs are short+isolated too
#         c) Boundary failure  — position offsets computed by double-counting
#                                  nested-tag text
#         d) Header miss       — tag-level scanning misses headers that are
#                                  plain text nodes inside a large parent tag
#   7. Text-node-level position truth + <a> exclusion + TOC scoring — fixed
#      all 4 classes (0 "failed" results across 40 tickers), but exposed
#      3 FURTHER classes on the needs_review tickers:
#         e) Combined headers   — O&G filers merge "Item 1 and 2. Business
#                                   and Properties" into one header; neither
#                                   the Item 1 nor Item 2 pattern matches it
#         f) Phrasing variance  — e.g. Item 5's subtitle varies by filer
#                                   ("Market for the Registrant's..." vs
#                                   "Market Information" vs "Market for
#                                   Common Stock") and a single rigid regex
#                                   missed several of them
#         g) Stub-header trap   — some filers (MCD, LOW, INTC pattern: EVERY
#                                   item ends up too_short in lockstep) have
#                                   more than one short isolated occurrence
#                                   of a header's text, and the "first
#                                   non-TOC, non-link" candidate picked is a
#                                   stub sitting right next to the NEXT
#                                   item's header, not the one followed by
#                                   real narrative content.
#
# CURRENT APPROACH (v8):
#   - Position truth: one ordered list of non-empty text nodes
#     (NavigableStrings), built once. No double counting — fixes (c).
#   - Header candidates collected at both text-node and tag level — fixes (d).
#   - Hyperlinked candidates (<a> tags / descendants of <a>) hard-excluded —
#     fixes (b), the dominant cause of cross-ref capture in modern filings.
#   - TOC candidates scored on 3 independent signals (in <table>, trailing
#     page-number digits, falls in first 8% of document) rather than one
#     fragile "prefer last" guess — fixes (a).
#   - ITEM_PATTERNS now holds a LIST of alternate regexes per item, so
#     filer-specific phrasing variance and combined Item 1/2 headers are
#     handled by adding alternates, not per-ticker patches — fixes (e), (f).
#   - After initial section extraction, any item whose section comes out
#     implausibly short (below its LENGTH_BOUNDS minimum) automatically
#     retries with the NEXT surviving candidate for that item (by document
#     order) instead of giving up — fixes (g). This also incidentally
#     repairs several order_correct failures, since picking the wrong stub
#     occurrence was what threw off relative ordering in those filings too.

# =============================================================================
# src/extraction/section_splitter.py
# =============================================================================
# Splits a raw SEC filing into named Item sections.
#
# BUG LOG (each round found genuinely new failure classes against real
# 40-ticker test data — documented so nobody re-introduces a fixed bug):
#
#   ROUND 1 (flattened-text heuristics, 5 attempts): all broke because a
#     cross-reference and a real header look identical as plain text but
#     are structurally different in the DOM.
#
#   ROUND 2 (isolated-DOM-element matching): fixed most cases, but exposed
#     4 generic classes: TOC capture, cross-ref capture (via short
#     hyperlinked anchor text), boundary failure (position offsets computed
#     by double-counting nested-tag text), header-detection miss (tag-level
#     scanning missed headers that are plain text nodes inside large parents).
#
#   ROUND 3 (text-node position truth + <a> exclusion + TOC scoring): fixed
#     all 4 of the above (0 "failed" across 40 tickers), but exposed 3 more:
#       e) combined headers ("Item 1 and 2. Business and Properties" in O&G
#          filers — neither single-item pattern matched it)
#       f) phrasing variance (Item 5's subtitle varies by filer)
#       g) stub-header trap (some filers have the header matched correctly
#          but the chosen occurrence is immediately followed by the NEXT
#          item's header, producing a near-zero-length section)
#
#   ROUND 4 (multi-pattern matching + naive single-pass retry-on-short):
#     fixed (f) almost entirely and (e) partially, but introduced a NEW bug:
#       h) combined-header double-claim — putting the combined pattern in
#          BOTH Item 1's and Item 2's pattern list means the same DOM node
#          gets claimed as the position for both items. They tie in sort
#          order, so whichever sorts first gets a zero-width slice and the
#          other swallows all the content meant for both.
#       i) order regression from the retry — accepting a "longer" candidate
#          for one item without checking whether that move broke relative
#          ordering against its neighbors. order_correct failures went
#          4 -> 14 as a direct result.
#
# ROUND 5 (this version) fixes (h) and (i) directly:
#   - The combined-header pattern now lives ONLY under Item 1. Item 2 is
#     back-filled post-hoc from Item 1's combined match (same text, no
#     separate position claim — eliminates the duplicate-position bug at
#     the source instead of guessing which one should "win" downstream).
#   - The stub-header retry is now ORDER-AWARE: a candidate substitution is
#     only kept if the resulting sequence is still order_correct (or was
#     already broken and this move doesn't make it worse). No more
#     accepting word-count improvements that silently wreck ordering.

# =============================================================================
# src/extraction/section_splitter.py  —  v9 (final)
# =============================================================================
# KNOWN FILING STRUCTURES (discovered empirically across 40 tickers):
#
#   A. STANDARD inline 10-K   — Item headers in body, prose follows inline.
#      Handled by: DOM text-node candidate search + rank/filter pipeline.
#
#   B. Incorporated-by-reference — body is a cross-reference sheet listing
#      page numbers in an attached Annual Report exhibit ("Item 1 Business
#      Pages 3-7, 9"). Content not in this document at all.
#      Detected by: PAGE_REFERENCE_PATTERN hitting ≥ 6 items.
#      Observed: MCD, LOW, INTC.
#
#   C. Combined Item 1 & 2 header — O&G filers write "Item 1 and 2.
#      Business and Properties" as one header. Items 1 and 2 are
#      extracted as a single combined section.
#      Observed: COP, PSX, OXY (and some of these ALSO incorporate by
#      reference, so Item 1 content is near-zero words).
#
#   D. Boundary-eaten sections — a header for Item N+1 was not found as
#      an isolated element, so Item N runs all the way to Item N+2's
#      header. Produces implausibly large sections (e.g. Item 7A = 47K
#      words). Repaired by: text-search fallback inside too_long sections.
#      Observed: MPC, PFE, BMY (Item 7A), XOM (Item 2).
#
# FAILURE CLASSES FIXED ACROSS ROUNDS (kept for institutional memory):
#   Round 2: TOC capture, cross-ref capture, boundary double-count, header miss
#   Round 3: Combined headers, Item 5 phrasing variance, stub-header trap
#   Round 4: Combined-header duplicate position claim, order-regression from retry
#   Round 5 (this): IbR detection, boundary repair, realistic length bounds,
#                   pattern whitespace tightening to kill concatenation artifacts

# =============================================================================
# src/extraction/section_splitter.py  —  v10
# =============================================================================
# KNOWN FILING STRUCTURES (empirically discovered across 40 tickers):
#
#   A. Standard inline 10-K — headers in body, prose follows.
#   B. Incorporated-by-reference (IbR) — body is a cross-reference sheet
#      pointing to a separate Annual Report exhibit. Content not in this file.
#      MCD, INTC: fully IbR (nothing extractable).
#      LOW: partially IbR — most items cross-referenced, Item 7 inline.
#   C. Combined Item 1 & 2 header — O&G filers: "Item 1 and 2. Business
#      and Properties" as one header.
#   D. Boundary-eaten sections — next header not found as isolated element,
#      so current section runs into the next. Repaired by text-search fallback.
#
# KEY REGRESSION FIXED in v10 vs v9:
#   v9 changed \s* → \s+ between item number and title to kill the MCD
#   concatenation artifact ("Item\xa01Business"). This broke EVERY filing
#   where adjacent inline elements are joined by get_text() without a space
#   (e.g. <b>ITEM 1.</b><b>BUSINESS</b> → "ITEM 1.BUSINESS"). TSLA, AMZN,
#   SBUX and dozens of others went from verified/needs_review to all-MISSING.
#
#   FIX: revert to \s* in patterns (necessary for robustness). Detect IbR
#   filings via POSITION: for real filings Item 1 appears at 5-20% of the
#   document; for IbR filings the cross-reference table sits at 85-99%.
#   If every candidate across all items is past 80% of the document,
#   return immediately as incorporated_by_reference.

import re
import json
import logging
import warnings
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

# ---------------------------------------------------------------------------
# Patterns — list per item, ANY-of matching.
# \s* between number and title is intentional: adjacent inline elements
# joined by get_text() may have no space ("ITEM 1.BUSINESS").
# The combined-header pattern lives ONLY under "1" — see build_sections().
# ---------------------------------------------------------------------------
_COMBINED_1_2 = [
    r"items?\s*1\s*(?:and|&)\s*2\.?\s*business",
    r"item\s*1\.?\s*business\s*and\s*properties",
]

ITEM_PATTERNS = {
    "1"  : [r"item\s*1\.?\s*business", *_COMBINED_1_2],
    "1A" : [r"item\s*1a\.?\s*risk\s*factors"],
    "1B" : [r"item\s*1b\.?\s*unresolved\s*staff"],
    "2"  : [r"item\s*2\.?\s*properties"],
    "3"  : [r"item\s*3\.?\s*legal\s*proceedings"],
    "5"  : [r"item\s*5\.?\s*market\s*for", r"item\s*5\.?\s*market\s*information"],
    "7"  : [r"item\s*7\.?\s*management"],
    "7A" : [r"item\s*7a\.?\s*quantitative"],
    "8"  : [r"item\s*8\.?\s*financial\s*statements"],
    "9"  : [r"item\s*9\.?\s*changes\s*in"],
    "9A" : [r"item\s*9a\.?\s*controls"],
    "10" : [r"item\s*10\.?\s*directors"],
}

ITEM_ORDER = ["1", "1A", "1B", "2", "3", "5", "7", "7A", "8", "9", "9A", "10"]

# Bounds calibrated against real 40-ticker data.
# Minimums are intentionally low for sections that can be one sentence.
# Maximums are the trigger for the boundary-repair pass.
LENGTH_BOUNDS = {
    "1"  : (200, 30000),
    "1A" : (500, 35000),
    "1B" : (0,   3000),
    "2"  : (10,  5000),
    "3"  : (10,  8000),
    "5"  : (20,  5000),
    "7"  : (200, 60000),
    "7A" : (10,  15000),
    "8"  : (10,  80000),
    "9"  : (0,   5000),
    "9A" : (10,  5000),
}

MAX_HEADER_ELEMENT_LENGTH = 200
TOC_REGION_FRACTION       = 0.08
TRAILING_PAGE_NUMBER      = re.compile(r"\d+\s*$")

# If an element's short text matches an item pattern but also contains
# an explicit "Pages N" phrase, it's a cross-reference sheet entry —
# record the hit but don't add it as an extractable candidate.
PAGE_REFERENCE_PATTERN = re.compile(r"\bpages?\s*\d", re.IGNORECASE)

# If ALL candidates across ALL items sit past this fraction of the document,
# the filing is incorporated-by-reference: content lives in a separate exhibit.
# Real filings have Item 1 at ~5-20%; IbR cross-ref tables sit at 85-99%.
IBR_POSITION_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matched_pattern(patterns, text):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p
    return None


def _is_cross_reference(node) -> bool:
    parent = node.parent if isinstance(node, NavigableString) else node
    if parent is None:
        return False
    return parent.name == "a" or parent.find_parent("a") is not None


def _is_in_table(node) -> bool:
    parent = node.parent if isinstance(node, NavigableString) else node
    return parent is not None and parent.find_parent("table") is not None


def build_text_node_index(soup: BeautifulSoup):
    nodes = [n for n in soup.find_all(string=True) if n.strip()]
    node_id_to_idx = {id(n): i for i, n in enumerate(nodes)}
    return nodes, node_id_to_idx


def _first_text_node_of(tag, node_id_to_idx):
    for s in tag.find_all(string=True):
        if s.strip() and id(s) in node_id_to_idx:
            return node_id_to_idx[id(s)]
    return None


def find_header_candidates(soup, nodes, node_id_to_idx):
    candidates_by_item  = {iid: [] for iid in ITEM_PATTERNS}
    page_reference_hits = {iid: 0  for iid in ITEM_PATTERNS}
    total_nodes         = len(nodes)
    seen_positions      = set()

    def consider(text_raw, node_idx, source_obj, tag_name):
        text = re.sub(r"\s+", " ", text_raw).strip()
        if not text or len(text) > MAX_HEADER_ELEMENT_LENGTH:
            return
        for item_id, patterns in ITEM_PATTERNS.items():
            matched = _matched_pattern(patterns, text)
            if not matched:
                continue
            if PAGE_REFERENCE_PATTERN.search(text):
                page_reference_hits[item_id] += 1
                continue
            key = (item_id, node_idx)
            if key in seen_positions:
                continue
            seen_positions.add(key)
            candidates_by_item[item_id].append({
                "text"               : text,
                "node_idx"           : node_idx,
                "tag_name"           : tag_name,
                "is_link"            : _is_cross_reference(source_obj),
                "in_table"           : _is_in_table(source_obj),
                "is_toc_region"      : (node_idx / total_nodes) < TOC_REGION_FRACTION
                                        if total_nodes else False,
                "has_trailing_digits": bool(TRAILING_PAGE_NUMBER.search(text)),
                "is_combined"        : item_id == "1" and matched in _COMBINED_1_2,
            })

    for s in nodes:
        consider(str(s), node_id_to_idx[id(s)], s, "#text")

    for tag in soup.find_all(True):
        text = tag.get_text(strip=True)
        if not text or len(text) > MAX_HEADER_ELEMENT_LENGTH:
            continue
        idx = _first_text_node_of(tag, node_id_to_idx)
        if idx is None:
            continue
        consider(text, idx, tag, tag.name)

    return candidates_by_item, page_reference_hits


def _rank_candidates(candidates):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: c["node_idx"])
    non_link   = [c for c in candidates if not c["is_link"]]
    pool       = non_link if non_link else candidates

    def suspicion(c):
        return (int(c["in_table"])
                + int(c["has_trailing_digits"])
                + int(c["is_toc_region"]))

    return sorted(pool, key=lambda c: (suspicion(c), c["node_idx"]))


def _repair_too_long_sections(sections: dict, is_combined_12: bool) -> dict:
    """
    If a section exceeds its max word bound, the next header was missed by
    the isolated-element search. Text-search inside the oversized section
    for the first occurrence of the next item's pattern that falls AFTER
    the approximate char position where legitimate content should have ended
    (max_words * 5 chars). This skips cross-references embedded in real
    content and lands on the actual structural header.
    """
    repaired = dict(sections)

    for i, item_id in enumerate(ITEM_ORDER):
        if item_id not in repaired or item_id == "10":
            continue
        if item_id == "2" and is_combined_12:
            continue

        bounds = LENGTH_BOUNDS.get(item_id)
        if not bounds:
            continue

        text       = repaired[item_id]
        word_count = len(text.split())
        if word_count <= bounds[1]:
            continue

        approx_max_chars = bounds[1] * 5
        cut_pos          = None
        rescued_id       = None

        for next_id in ITEM_ORDER[i + 1:]:
            if next_id == "10":
                continue
            late, early = [], []
            for pattern in ITEM_PATTERNS.get(next_id, []):
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    if m.start() <= 100:
                        continue
                    (late if m.start() >= approx_max_chars else early).append(m.start())

            if late:
                cut_pos    = min(late)
                rescued_id = next_id
                break
            if early:
                cut_pos    = max(early)
                rescued_id = next_id

        if cut_pos is not None:
            repaired[item_id] = text[:cut_pos].strip()
            remaining         = text[cut_pos:].strip()
            if rescued_id and (
                rescued_id not in repaired
                or len(repaired[rescued_id].split()) < 50
            ):
                repaired[rescued_id] = remaining

    return repaired


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def split_filing(raw_html: str, ticker: str = "", filing_date: str = "") -> dict:
    soup = BeautifulSoup(raw_html, "lxml")
    nodes, node_id_to_idx = build_text_node_index(soup)
    total_nodes           = len(nodes)

    candidates_by_item, page_reference_hits = find_header_candidates(
        soup, nodes, node_id_to_idx
    )

    # ── IbR detection via candidate position ────────────────────────────
    # Collect the best (lowest suspicion, earliest) candidate position for
    # each item that has any candidate at all.
    all_best_positions = []
    for item_id, cands in candidates_by_item.items():
        ranked = _rank_candidates(cands)
        if ranked:
            all_best_positions.append(ranked[0]["node_idx"])

    if total_nodes > 0 and all_best_positions:
        earliest_fraction = min(all_best_positions) / total_nodes
        if earliest_fraction > IBR_POSITION_THRESHOLD:
            # Every matched header is deep in the document — this is a
            # cross-reference sheet pointing to a separate exhibit.
            return {
                "ticker"      : ticker,
                "filing_date" : filing_date,
                "sections"    : {},
                "confidence"  : {
                    "overall_status"    : "incorporated_by_reference",
                    "earliest_match_pct": round(earliest_fraction * 100, 1),
                    "ibr_page_ref_hits" : {k: v for k, v in page_reference_hits.items() if v > 0},
                    "occurrence_counts" : {k: len(v) for k, v in candidates_by_item.items()},
                    "zero_occurrence"   : [],
                    "found_order"       : [],
                    "order_correct"     : True,
                    "length_check"      : {},
                    "length_failures"   : [],
                    "is_combined_1_2"   : False,
                },
            }

    occurrence_counts = {k: len(v) for k, v in candidates_by_item.items()}

    ranked_by_item = {
        iid: _rank_candidates(cands)
        for iid, cands in candidates_by_item.items()
    }
    chosen_rank_idx = {iid: 0 for iid, r in ranked_by_item.items() if r}

    def build_sections(rank_idx_map):
        real_headers = {}
        for iid, ranked in ranked_by_item.items():
            i = rank_idx_map.get(iid, 0)
            if ranked and i < len(ranked):
                real_headers[iid] = ranked[i]

        positions    = {iid: info["node_idx"] for iid, info in real_headers.items()}
        sorted_items = sorted(positions.items(), key=lambda x: x[1])
        node_texts   = [str(n).strip() for n in nodes]

        sections      = {}
        combined_flag = False
        for i, (iid, start_idx) in enumerate(sorted_items):
            if iid == "10":
                continue
            end_idx      = sorted_items[i + 1][1] if i + 1 < len(sorted_items) else len(nodes)
            text         = " ".join(t for t in node_texts[start_idx:end_idx] if t)
            sections[iid] = re.sub(r"\s+", " ", text).strip()
            if iid == "1" and real_headers.get("1", {}).get("is_combined"):
                combined_flag = True

        if combined_flag and "2" not in sections and "1" in sections:
            sections["2"] = sections["1"]

        return sections, sorted_items, combined_flag

    sections, sorted_items, is_combined_12 = build_sections(chosen_rank_idx)

    def _order_ok(si):
        found    = [i for i, _ in si if i != "10"]
        expected = [i for i in ITEM_ORDER if i in found]
        return found == expected

    baseline_order_ok = _order_ok(sorted_items)

    # ── Stub-header retry (order-aware) ─────────────────────────────────
    MAX_RETRIES = 4
    for item_id in [i for i in ITEM_ORDER if i in sections]:
        ranked = ranked_by_item.get(item_id, [])
        bounds = LENGTH_BOUNDS.get(item_id)
        if not bounds or len(ranked) <= 1:
            continue
        if item_id == "2" and is_combined_12:
            continue

        attempt = chosen_rank_idx.get(item_id, 0)
        tries   = 0
        while (len(sections[item_id].split()) < bounds[0]
               and attempt + 1 < len(ranked)
               and tries < MAX_RETRIES):
            attempt += 1
            tries   += 1
            trial_map = dict(chosen_rank_idx)
            trial_map[item_id] = attempt
            ts, tsi, tc = build_sections(trial_map)
            if (len(ts.get(item_id, "").split()) >= bounds[0]
                    and (_order_ok(tsi) or not baseline_order_ok)):
                chosen_rank_idx[item_id] = attempt
                sections, sorted_items, is_combined_12 = ts, tsi, tc
                baseline_order_ok = _order_ok(sorted_items)
                break

    sections, sorted_items, is_combined_12 = build_sections(chosen_rank_idx)

    # ── Boundary repair for too_long sections ───────────────────────────
    sections = _repair_too_long_sections(sections, is_combined_12)

    # ── Verification ────────────────────────────────────────────────────
    found_order          = [iid for iid, _ in sorted_items if iid != "10"]
    expected_subsequence = [i for i in ITEM_ORDER if i in found_order]
    order_correct        = (found_order == expected_subsequence)

    length_check = {}
    for iid, text in sections.items():
        if iid == "2" and is_combined_12:
            length_check[iid] = "combined_with_item_1"
            continue
        wc     = len(text.split())
        bounds = LENGTH_BOUNDS.get(iid)
        if bounds is None:
            length_check[iid] = "no_bounds_defined"
        elif wc < bounds[0]:
            length_check[iid] = f"too_short ({wc}w, expected>={bounds[0]})"
        elif wc > bounds[1]:
            length_check[iid] = f"too_long ({wc}w, expected<={bounds[1]})"
        else:
            length_check[iid] = "ok"

    length_failures = [
        k for k, v in length_check.items()
        if v not in ("ok", "no_bounds_defined", "combined_with_item_1")
    ]
    zero_occurrence = [k for k, v in occurrence_counts.items() if v == 0]

    overall_status = "verified"
    if zero_occurrence or not order_correct or length_failures:
        overall_status = "needs_review"
    if not sections:
        overall_status = "failed"

    return {
        "ticker"      : ticker,
        "filing_date" : filing_date,
        "sections"    : sections,
        "confidence"  : {
            "occurrence_counts" : occurrence_counts,
            "zero_occurrence"   : zero_occurrence,
            "found_order"       : found_order,
            "order_correct"     : order_correct,
            "length_check"      : length_check,
            "length_failures"   : length_failures,
            "is_combined_1_2"   : is_combined_12,
            "overall_status"    : overall_status,
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    test_path  = Path("data/raw/filings/AAPL")
    json_files = list(test_path.glob("10K_*.json"))
    if not json_files:
        print("No AAPL 10-K found"); sys.exit(1)

    raw    = json.load(open(json_files[0]))
    result = split_filing(raw["raw_html"], ticker="AAPL", filing_date=raw["filing_date"])

    print("=" * 70)
    print(json.dumps(result["confidence"], indent=2))
    print()
    for iid, text in result["sections"].items():
        print(f"--- Item {iid} ({len(text.split())} words) ---")
        print(text[:200] + "...")
        print()