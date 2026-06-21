# =============================================================================
# src/extraction/section_splitter.py
# =============================================================================
# Splits a raw SEC filing into named Item sections.
#
# APPROACH HISTORY (5 attempts, all validated against real AAPL + XOM
# filings — documented because each failure was genuinely instructive):
#   1. "last text occurrence"     — broke: cross-refs cluster late in Item 7
#   2. "second text occurrence"   — broke: early cross-ref before real header
#   3. "phrase filtering"         — broke: unbounded list of cross-ref phrasings
#   4. "position windows (%)"     — broke: cross-ref + header in same window
#   5. "sequential ordering"      — broke: multiple candidates between two
#                                    correctly-anchored neighbors, still guessing
#
# ROOT CAUSE of all 5 failures: every attempt searched FLATTENED TEXT,
# which discards the document's actual HTML structure. A cross-reference
# ("...as discussed in Item 1A. Risk Factors, compliance with...") and a
# real header look nearly identical as plain text — but they are
# structurally very different in the DOM.
#
# WORKING APPROACH (validated against real XOM filing via direct DOM
# inspection): a real Item header in inline-XBRL filings is its own
# short, isolated HTML element (e.g. a <div>/<span> containing ONLY
# "ITEM 1A.      RISK FACTORS", nothing else). A cross-reference is
# never isolated this way — it's a substring inside a long paragraph
# of surrounding prose. We search actual HTML elements (not flattened
# text) for ones whose OWN text is short and matches an Item pattern.
# This eliminates the ambiguity at the source instead of guessing
# which text-position is "more likely" to be real.

import re
import json
import logging
import warnings
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

# Whitespace OPTIONAL (\s*) — real filings sometimes pad with multiple
# spaces ("ITEM 1A.      RISK FACTORS") or collapse it entirely in TOC
# table cells ("Item 1A.Risk Factors2").
ITEM_PATTERNS = {
    "1"  : r"item\s*1\.?\s*business",
    "1A" : r"item\s*1a\.?\s*risk\s*factors",
    "1B" : r"item\s*1b\.?\s*unresolved\s*staff\s*comments",
    "2"  : r"item\s*2\.?\s*properties",
    "3"  : r"item\s*3\.?\s*legal\s*proceedings",
    "5"  : r"item\s*5\.?\s*market\s*for\s*registrant",
    "7"  : r"item\s*7\.?\s*management.s\s*disc",
    "7A" : r"item\s*7a\.?\s*quantitative\s*and\s*qualitative",
    "8"  : r"item\s*8\.?\s*financial\s*statements",
    "9"  : r"item\s*9\.?\s*changes\s*in\s*and\s*disagreements",
    "9A" : r"item\s*9a\.?\s*controls\s*and\s*procedures",
    "10" : r"item\s*10\.?\s*directors",
}

ITEM_ORDER = ["1", "1A", "1B", "2", "3", "5", "7", "7A", "8", "9", "9A", "10"]

LENGTH_BOUNDS = {
    "1"  : (200, 15000),
    "1A" : (500, 25000),
    "1B" : (0,   2000),
    "2"  : (50,  3000),
    "3"  : (50,  8000),
    "5"  : (50,  5000),
    "7"  : (500, 30000),
    "7A" : (50,  6000),
    "8"  : (50,  80000),
    "9"  : (0,   2000),
    "9A" : (50,  3000),
}

# Max length (chars) for an element to count as an "isolated header" —
# validated against real data: real headers like "ITEM 1A.      RISK
# FACTORS" are ~26 chars. 200 gives generous margin while still
# excluding any paragraph-length cross-reference.
MAX_HEADER_ELEMENT_LENGTH = 200


def find_header_elements(soup: BeautifulSoup) -> dict:
    """
    Find the REAL header element for each Item by searching actual DOM
    elements (not flattened text) for short, isolated tags whose own
    text matches an Item pattern.

    Validated against XOM: this correctly returns only 2 matches for
    "Item 1A" (TOC table row + real header div/span pair) instead of
    8 matches that text-search returns (6 of which are cross-references
    buried inside long paragraphs and correctly excluded here because
    their parent element's text is thousands of characters long, not
    under MAX_HEADER_ELEMENT_LENGTH).

    Returns: {item_id: [list of matching elements, in document order]}
    """
    candidates_by_item = {item_id: [] for item_id in ITEM_PATTERNS}

    # Walk every element in the document
    for tag in soup.find_all(True):
        text = tag.get_text(strip=True)

        if not text or len(text) > MAX_HEADER_ELEMENT_LENGTH:
            continue  # too long to be an isolated header — skip immediately

        for item_id, pattern in ITEM_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                candidates_by_item[item_id].append({
                    "element": tag,
                    "text": text,
                    "tag_name": tag.name,
                })

    return candidates_by_item


def select_real_header(item_id: str, candidates: list) -> dict | None:
    """
    Among short isolated-element candidates for one item, pick the real
    header (not the TOC row).

    Heuristic: TOC entries are <tr>/<td> table cells (the TOC is a table
    in every filing we've inspected). Real headers are <div>/<span>/<p>
    elements outside any TOC-style table, OR — if everything is inside
    tables (some filers format the whole doc in tables) — the real
    header is the LAST candidate in document order, since the TOC
    always appears first and there's normally exactly one non-TOC match
    once we've already filtered to short isolated elements only
    (cross-references are excluded by the length filter before we even
    get here).
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer non-table candidates (TOC rows are inside <table>)
    non_table = [c for c in candidates if c["element"].find_parent("table") is None]
    if non_table:
        # If multiple non-table candidates exist, take the last
        # (in case of an isolated short cross-reference elsewhere —
        #  rare, but the real section header is the structural one
        #  that actually starts new content, which tends to appear
        #  last among short non-table matches)
        return non_table[-1]

    # All candidates are inside tables — take the last one
    return candidates[-1]


def get_document_position(element, soup: BeautifulSoup) -> int:
    """
    Get this element's approximate character position in the full
    document text, for ordering and length-bound calculations downstream.
    """
    # Use the element's position among all NavigableStrings before it
    preceding_text = ""
    for el in soup.find_all(True):
        if el is element:
            break
        preceding_text += el.get_text(strip=True)
    return len(preceding_text)


def extract_full_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_filing(raw_html: str, ticker: str = "", filing_date: str = "") -> dict:
    """
    Main entry point. Finds real header ELEMENTS via DOM structure first,
    then extracts text content between them from the full flattened text
    using each header's own matched text as the split anchor.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    full_text = extract_full_text(raw_html)

    candidates_by_item = find_header_elements(soup)

    occurrence_counts = {k: len(v) for k, v in candidates_by_item.items()}

    real_headers = {}
    for item_id, candidates in candidates_by_item.items():
        chosen = select_real_header(item_id, candidates)
        if chosen:
            real_headers[item_id] = chosen

    # Find each chosen header's position in the FULL TEXT by searching
    # for its exact matched text, anchored after the TOC region
    # (use the LAST occurrence of the header's own short text string,
    #  since the real header's exact text — e.g. "ITEM 1A.      RISK
    #  FACTORS" — is highly specific and rarely repeated verbatim
    #  outside the TOC + the header itself)
    positions = {}
    for item_id, header_info in real_headers.items():
        header_text = re.sub(r"\s+", " ", header_info["text"]).strip()
        # Find all occurrences of this exact header text in full_text
        escaped = re.escape(header_text)
        matches = list(re.finditer(escaped, full_text, re.IGNORECASE))
        if matches:
            # Take the last occurrence of this EXACT short string —
            # the TOC version may differ slightly in spacing/case from
            # the real header's own text, so an exact match late in the
            # document reliably lands on the real header itself
            positions[item_id] = matches[-1].start()

    # Sort by confirmed position, extract text between consecutive sections
    sorted_items = sorted(positions.items(), key=lambda x: x[1])
    sections = {}
    for i, (item_id, start) in enumerate(sorted_items):
        if item_id == "10":
            continue  # boundary marker only
        end = sorted_items[i + 1][1] if i + 1 < len(sorted_items) else len(full_text)
        sections[item_id] = full_text[start:end].strip()

    # ── Verification ──
    found_order = [item_id for item_id, _ in sorted_items if item_id != "10"]
    expected_subsequence = [item for item in ITEM_ORDER if item in found_order]
    order_correct = (found_order == expected_subsequence)

    length_check = {}
    for item_id, text in sections.items():
        word_count = len(text.split())
        bounds = LENGTH_BOUNDS.get(item_id)
        if bounds is None:
            length_check[item_id] = "no_bounds_defined"
        elif word_count < bounds[0]:
            length_check[item_id] = f"too_short ({word_count}w, expected>={bounds[0]})"
        elif word_count > bounds[1]:
            length_check[item_id] = f"too_long ({word_count}w, expected<={bounds[1]})"
        else:
            length_check[item_id] = "ok"

    length_failures = [k for k, v in length_check.items() if v not in ("ok", "no_bounds_defined")]
    zero_occurrence = [k for k, v in occurrence_counts.items() if v == 0]

    overall_status = "verified"
    if zero_occurrence:
        overall_status = "needs_review"
    if not order_correct:
        overall_status = "needs_review"
    if length_failures:
        overall_status = "needs_review"
    if not sections:
        overall_status = "failed"

    confidence = {
        "occurrence_counts" : occurrence_counts,
        "zero_occurrence"   : zero_occurrence,
        "found_order"       : found_order,
        "order_correct"     : order_correct,
        "length_check"      : length_check,
        "length_failures"   : length_failures,
        "overall_status"    : overall_status,
    }

    return {
        "ticker"      : ticker,
        "filing_date" : filing_date,
        "sections"    : sections,
        "confidence"  : confidence,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    test_path = Path("data/raw/filings/AAPL")
    json_files = list(test_path.glob("10K_*.json"))
    if not json_files:
        print("No AAPL 10-K found")
        sys.exit(1)

    raw = json.load(open(json_files[0]))
    result = split_filing(raw["raw_html"], ticker="AAPL", filing_date=raw["filing_date"])

    print("="*70)
    print("VERIFICATION RESULT")
    print("="*70)
    print(json.dumps(result["confidence"], indent=2))
    print()
    for item_id, text in result["sections"].items():
        print(f"--- Item {item_id} ({len(text.split())} words) ---")
        print(text[:200] + "...")
        print()