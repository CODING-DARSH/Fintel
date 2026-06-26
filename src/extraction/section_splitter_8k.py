# =============================================================================
# src/extraction/section_splitter_8k.py
# =============================================================================
# Splits a raw SEC 8-K filing into named Item sections.
#
# 8-Ks use decimal item numbering (1.01, 2.02, 5.02, 9.01) completely
# different from 10-K integer numbering (1, 1A, 1B...).
# Each 8-K only contains the items relevant to that specific event —
# a typical 8-K has 1-3 items, not the full set.
#
# ITEM TAXONOMY:
#   1.01  Entry into Material Definitive Agreement
#   1.02  Termination of Material Definitive Agreement
#   1.03  Bankruptcy or Receivership
#   2.01  Completion of Acquisition or Disposition
#   2.02  Results of Operations (earnings)
#   2.03  Creation of Direct Financial Obligation
#   2.04  Triggering Events on Financial Obligation
#   2.05  Departure/Election of Directors or Officers
#   2.06  Amendments to Articles of Incorporation
#   3.01  Notice of Delisting
#   3.02  Unregistered Sales of Equity Securities
#   4.01  Changes in Certifying Accountant
#   5.01  Changes in Control of Registrant
#   5.02  Departure/Appointment of Officers (executive changes)
#   5.03  Amendments to Articles / Change in Fiscal Year
#   5.07  Submission of Matters to Vote
#   6.01  ABS Informational
#   7.01  Regulation FD Disclosure
#   8.01  Other Events
#   9.01  Financial Statements and Exhibits
#
# APPROACH: same text-node index approach as section_splitter.py —
# build one ordered list of NavigableStrings for position truth,
# find isolated header elements matching decimal item patterns,
# filter TOC/cross-refs, extract text between consecutive headers.

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
# 8-K item patterns — decimal notation
# ---------------------------------------------------------------------------
ITEM_8K_PATTERNS = {
    "1.01": [r"item\s*1\.01[\s\.\-]*(?:entry\s*into|material\s*definitive)"],
    "1.02": [r"item\s*1\.02[\s\.\-]*termination"],
    "1.03": [r"item\s*1\.03[\s\.\-]*bankruptcy"],
    "2.01": [r"item\s*2\.01[\s\.\-]*(?:completion|acquisition|disposition)"],
    "2.02": [r"item\s*2\.02[\s\.\-]*results\s*of\s*operations"],
    "2.03": [r"item\s*2\.03[\s\.\-]*(?:creation|direct\s*financial)"],
    "2.04": [r"item\s*2\.04[\s\.\-]*triggering"],
    "2.05": [r"item\s*2\.05[\s\.\-]*(?:departure|directors|officers)"],
    "2.06": [r"item\s*2\.06[\s\.\-]*amendments"],
    "3.01": [r"item\s*3\.01[\s\.\-]*(?:delisting|notice)"],
    "3.02": [r"item\s*3\.02[\s\.\-]*unregistered"],
    "4.01": [r"item\s*4\.01[\s\.\-]*(?:changes|accountant)"],
    "5.01": [r"item\s*5\.01[\s\.\-]*changes\s*in\s*control"],
    "5.02": [r"item\s*5\.02[\s\.\-]*(?:departure|appointment|officers|directors)"],
    "5.03": [r"item\s*5\.03[\s\.\-]*(?:amendments|fiscal\s*year)"],
    "5.07": [r"item\s*5\.07[\s\.\-]*(?:submission|vote)"],
    "6.01": [r"item\s*6\.01[\s\.\-]*abs"],
    "7.01": [r"item\s*7\.01[\s\.\-]*(?:regulation\s*fd|reg\s*fd)"],
    "8.01": [r"item\s*8\.01[\s\.\-]*other\s*events"],
    "9.01": [r"item\s*9\.01[\s\.\-]*(?:financial\s*statements|exhibits)"],
}

# Broad fallback pattern — catches any "Item X.XX" header even if title
# doesn't match our known taxonomy (filers sometimes use non-standard wording)
GENERIC_8K_PATTERN = re.compile(
    r"^item\s*(\d+\.\d+)", re.IGNORECASE
)

ITEM_8K_ORDER = [
    "1.01","1.02","1.03",
    "2.01","2.02","2.03","2.04","2.05","2.06",
    "3.01","3.02",
    "4.01",
    "5.01","5.02","5.03","5.07",
    "6.01",
    "7.01",
    "8.01",
    "9.01",
]

# Human-readable labels for reporting
ITEM_8K_LABELS = {
    "1.01": "Entry into Material Agreement",
    "1.02": "Termination of Material Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition/Disposition",
    "2.02": "Results of Operations (Earnings)",
    "2.03": "Creation of Financial Obligation",
    "2.04": "Triggering Event on Obligation",
    "2.05": "Departure/Election of Directors/Officers",
    "2.06": "Amendments to Articles",
    "3.01": "Notice of Delisting",
    "3.02": "Unregistered Sales of Equity",
    "4.01": "Change in Certifying Accountant",
    "5.01": "Change in Control",
    "5.02": "Departure/Appointment of Officers",
    "5.03": "Amendments / Fiscal Year Change",
    "5.07": "Submission to Shareholder Vote",
    "6.01": "ABS Informational",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

MAX_HEADER_ELEMENT_LENGTH = 300   # 8-K headers can be slightly longer
TOC_REGION_FRACTION       = 0.10
TRAILING_PAGE_NUMBER      = re.compile(r"\d+\s*$")


# ---------------------------------------------------------------------------
# Shared utilities (same approach as section_splitter.py)
# ---------------------------------------------------------------------------

def _matched_pattern_8k(text):
    """
    Returns (item_id, matched_pattern) for the first known item pattern
    that matches, or tries the generic fallback.
    """
    for item_id, patterns in ITEM_8K_PATTERNS.items():
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return item_id, p
    # Generic fallback — catches items we haven't explicitly patterned
    m = GENERIC_8K_PATTERN.search(text)
    if m:
        item_id = m.group(1)
        return item_id, "generic"
    return None, None


def _is_cross_reference(node) -> bool:
    parent = node.parent if isinstance(node, NavigableString) else node
    if parent is None:
        return False
    return parent.name == "a" or parent.find_parent("a") is not None


def _is_in_table(node) -> bool:
    parent = node.parent if isinstance(node, NavigableString) else node
    return parent is not None and parent.find_parent("table") is not None


def build_text_node_index(soup):
    nodes = [n for n in soup.find_all(string=True) if n.strip()]
    node_id_to_idx = {id(n): i for i, n in enumerate(nodes)}
    return nodes, node_id_to_idx


def _first_text_node_of(tag, node_id_to_idx):
    for s in tag.find_all(string=True):
        if s.strip() and id(s) in node_id_to_idx:
            return node_id_to_idx[id(s)]
    return None


def find_8k_candidates(soup, nodes, node_id_to_idx):
    candidates_by_item = {}
    total_nodes        = len(nodes)
    seen_positions     = set()

    def consider(text_raw, node_idx, source_obj, tag_name):
        text = re.sub(r"\s+", " ", text_raw).strip()
        if not text or len(text) > MAX_HEADER_ELEMENT_LENGTH:
            return
        item_id, matched = _matched_pattern_8k(text)
        if not item_id:
            return
        key = (item_id, node_idx)
        if key in seen_positions:
            return
        seen_positions.add(key)
        if item_id not in candidates_by_item:
            candidates_by_item[item_id] = []
        candidates_by_item[item_id].append({
            "text"               : text,
            "node_idx"           : node_idx,
            "tag_name"           : tag_name,
            "matched"            : matched,
            "is_link"            : _is_cross_reference(source_obj),
            "in_table"           : _is_in_table(source_obj),
            "is_toc_region"      : (node_idx / total_nodes) < TOC_REGION_FRACTION
                                    if total_nodes else False,
            "has_trailing_digits": bool(TRAILING_PAGE_NUMBER.search(text)),
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

    return candidates_by_item


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def split_8k(raw_html: str, ticker: str = "", filing_date: str = "") -> dict:
    """
    Split an 8-K filing into its Item sections.

    Returns:
        ticker, filing_date, sections {item_id: text},
        confidence {items_found, occurrence_counts, overall_status}
    """
    soup = BeautifulSoup(raw_html, "lxml")
    nodes, node_id_to_idx = build_text_node_index(soup)

    candidates_by_item = find_8k_candidates(soup, nodes, node_id_to_idx)
    occurrence_counts  = {k: len(v) for k, v in candidates_by_item.items()}

    ranked_by_item = {
        iid: _rank_candidates(cands)
        for iid, cands in candidates_by_item.items()
    }

    # Pick best candidate per item
    real_headers = {}
    for iid, ranked in ranked_by_item.items():
        if ranked:
            real_headers[iid] = ranked[0]

    if not real_headers:
        return {
            "ticker"      : ticker,
            "filing_date" : filing_date,
            "sections"    : {},
            "confidence"  : {
                "overall_status"  : "failed",
                "items_found"     : [],
                "occurrence_counts": occurrence_counts,
            },
        }

    # Sort by position, extract text between consecutive headers
    positions    = {iid: info["node_idx"] for iid, info in real_headers.items()}
    sorted_items = sorted(positions.items(), key=lambda x: x[1])
    node_texts   = [str(n).strip() for n in nodes]

    sections = {}
    for i, (iid, start_idx) in enumerate(sorted_items):
        end_idx = sorted_items[i + 1][1] if i + 1 < len(sorted_items) else len(nodes)
        text    = " ".join(t for t in node_texts[start_idx:end_idx] if t)
        sections[iid] = re.sub(r"\s+", " ", text).strip()

    items_found = sorted(sections.keys())

    # Classify event types present
    event_types = {
        iid: ITEM_8K_LABELS.get(iid, f"Item {iid}")
        for iid in items_found
    }

    # Overall status
    if not sections:
        overall_status = "failed"
    elif len(sections) == 1 and "9.01" in sections:
        overall_status = "exhibits_only"  # just an exhibits filing, no narrative
    else:
        overall_status = "extracted"

    return {
        "ticker"      : ticker,
        "filing_date" : filing_date,
        "sections"    : sections,
        "confidence"  : {
            "overall_status"   : overall_status,
            "items_found"      : items_found,
            "event_types"      : event_types,
            "occurrence_counts": occurrence_counts,
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    test_path  = Path("data/raw/filings/AAPL")
    json_files = [f for f in test_path.glob("8K_*.json")]
    if not json_files:
        print("No AAPL 8-K found"); sys.exit(1)

    raw    = json.load(open(json_files[0]))
    result = split_8k(raw["raw_html"], ticker="AAPL", filing_date=raw["filing_date"])

    print("=" * 70)
    print(json.dumps(result["confidence"], indent=2))
    print()
    for iid, text in result["sections"].items():
        label = ITEM_8K_LABELS.get(iid, f"Item {iid}")
        print(f"--- Item {iid} | {label} ({len(text.split())} words) ---")
        print(text[:300] + "...")
        print()