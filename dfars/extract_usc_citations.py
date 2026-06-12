"""Regex-extract U.S. Code citations from each DFARS title-48 XML version.

For every `data/DFARS/title-48_<date>.xml` file, write a matching
`data/DFARS/usc_citations_<date>.json` with two top-level indexes:

    {
      "section_to_citations": {
        "225.7002": [
          {"cite": "10 U.S.C. 4862(k)", "section": "10 U.S.C. 4862", "raw": "10 U.S.C. 4862(k)"},
          ...
        ]
      },
      "citation_to_sections": {
        "10 U.S.C. 4862": ["225.7002", ...]
      }
    }

`section_to_citations` is keyed by DFARS node number (the same `N` identifiers
used by `extract_hierarchy.py`); each value is the list of distinct USC citations
found in that node's own text. `citation_to_sections` is the reverse lookup —
keyed by the bare section-level USC citation, each value the sorted list of DFARS
nodes that cite it — so "which DFARS sections cite a given USC section?" is one
dictionary access.

Within a citation entry, `cite` is the citation as written (subsection parens and
`note` kept), `section` is normalized to the bare section level (parens/`note`
stripped) so it lines up with the section-level USC citations the NDAA extractor
produces — and is the key used in the reverse index — and `raw` is the full
matched span the item came from.

The extractor handles the list forms that appear in the data, e.g.
`10 U.S.C. 7504, 8354 and 3253` expands to three separate citations, while
refusing to swallow the title of a *following* citation: in
`41 U.S.C. 1303 and 48 CFR chapter 1` only `41 U.S.C. 1303` is captured.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from dfars.extract_hierarchy import natural_key

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "DFARS"

# DIV TYPEs whose `N` identifier should own the text directly inside them. These
# match the hierarchy nodes built by extract_hierarchy.py; there is no separate
# subsection type — subsections (e.g. 252.203-7000) are themselves TYPE="SECTION"
# with the full hyphenated number as their `N`.
TEXT_OWNING_TYPES = {"PART", "SUBPART", "SECTION"}

# A single section item: 2279, 2306a, 98h-1, 8502-8504, 40102(a)(4), 3204 note.
# `(?!\d)` after each integer stops greedy `\d+` from backtracking into a partial
# number (e.g. matching "4" out of "48 CFR ...") to satisfy a later guard.
_ITEM = (
    r"\d+(?!\d)[a-z]*(?:[-–]\d+(?!\d)[a-z]*)?"
    r"(?:\([^()\s]{1,6}\))*(?:\s+notes?(?:\s+prec\.)?)?"
)
# A continuation item must not itself be the title of a *new* citation, i.e. it
# is not a number immediately followed by "U.S.C.", "CFR", or "Stat".
_CONT = rf"{_ITEM}(?!\s*(?:U\.?\s?S\.?\s?C|CFR|Stat))"
# Separators between items in a list: commas and/or "and"/"or". The conjunction
# form is tried first so an Oxford comma (", and ") is consumed whole rather than
# leaving a stray "and 9540" token behind the plain-comma alternative.
_SEP = r",?\s+(?:and|or)\s+|\s*,\s*"

USC_RE = re.compile(
    rf"(?P<title>\d+)\s+U\.?\s?S\.?\s?C\.?\s+"
    rf"(?P<body>(?:chapters?\s+)?{_ITEM}(?:(?:{_SEP}){_CONT})*)"
)
PROSE_RE = re.compile(
    r"[Ss]ection\s+(?P<section>\d+[a-z]*(?:\([^()\s]{1,6}\))*)\s+of\s+[Tt]itle\s+(?P<title>\d+)\b"
)
_SPLIT_RE = re.compile(_SEP)
_NOTE_RE = re.compile(r"\s+notes?(?:\s+prec\.)?\s*$")
_PAREN_RE = re.compile(r"\([^()\s]{1,6}\)")


def _bare_section(item: str) -> str:
    """Normalize an item to the bare section level.

    Strips subsection parens and a trailing `note`, but keeps letter suffixes
    (`2306a`), ranges (`8502-8504`), and `chapter` items as written.
    """
    item = item.strip()
    if item.lower().startswith("chapter"):
        return item
    item = _NOTE_RE.sub("", item)
    item = _PAREN_RE.sub("", item)
    return item.strip()


def _emit(title: str, item: str, raw: str) -> dict:
    item = item.strip()
    return {
        "cite": f"{title} U.S.C. {item}",
        "section": f"{title} U.S.C. {_bare_section(item)}",
        "raw": raw,
    }


def extract_usc_citations(text: str) -> list[dict]:
    """Extract USC citations from a block of text, expanding list forms."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(entry: dict) -> None:
        key = (entry["cite"], entry["section"])
        if key not in seen:
            seen.add(key)
            out.append(entry)

    for m in USC_RE.finditer(text):
        title = m.group("title")
        body = m.group("body").strip()
        raw = m.group(0).strip()
        if body.lower().startswith("chapter"):
            # Chapter bodies are never lists in the data; keep whole.
            add(_emit(title, body, raw))
            continue
        for item in _SPLIT_RE.split(body):
            if item.strip():
                add(_emit(title, item, raw))

    for m in PROSE_RE.finditer(text):
        add(_emit(m.group("title"), m.group("section"), m.group(0).strip()))

    return out


def node_texts(path: Path) -> dict[str, str]:
    """Map each DFARS node number to the text that sits directly inside it.

    Text is attributed to the nearest enclosing PART/SUBPART/SECTION/SUBSECT node,
    so part-level front matter (e.g. an authority note) lands under the PART rather
    than its first section.
    """
    texts: dict[str, list[str]] = {}

    def add(key: str | None, s: str | None) -> None:
        if key is not None and s and s.strip():
            texts.setdefault(key, []).append(s)

    def walk(elem: ET.Element, current_key: str | None) -> None:
        if elem.attrib.get("TYPE") in TEXT_OWNING_TYPES and elem.attrib.get("N"):
            current_key = elem.attrib["N"]
        add(current_key, elem.text)
        for child in elem:
            walk(child, current_key)
            add(current_key, child.tail)

    root = ET.parse(path).getroot()
    walk(root, None)
    return {k: " ".join(v) for k, v in texts.items()}


def build_reverse_index(section_to_citations: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Invert the per-node citations into a section-level -> DFARS nodes map.

    Keyed by the bare `section` form (so `4862(k)` and `4862` collapse together),
    each value is the sorted, deduped list of DFARS nodes that cite it.
    """
    reverse: dict[str, set[str]] = {}
    for number, cites in section_to_citations.items():
        for entry in cites:
            reverse.setdefault(entry["section"], set()).add(number)
    return {
        section: sorted(nodes, key=natural_key)
        for section, nodes in sorted(reverse.items())
    }


def parse_file(path: Path) -> dict[str, dict]:
    section_to_citations: dict[str, list[dict]] = {}
    for number, text in node_texts(path).items():
        cites = extract_usc_citations(text)
        if cites:
            section_to_citations[number] = cites
    return {
        "section_to_citations": section_to_citations,
        "citation_to_sections": build_reverse_index(section_to_citations),
    }


def selftest() -> None:
    def cites(text: str) -> list[str]:
        return [e["cite"] for e in extract_usc_citations(text)]

    def sections(text: str) -> list[str]:
        return [e["section"] for e in extract_usc_citations(text)]

    # The user's motivating case: a three-section list expands to three.
    assert cites("10 U.S.C. 7504, 8354 and 3253") == [
        "10 U.S.C. 7504",
        "10 U.S.C. 8354",
        "10 U.S.C. 3253",
    ]
    # Oxford comma before the final "and" (real case: DFARS 236.606-70).
    assert cites("10 U.S.C. 7540, 8612, and 9540") == [
        "10 U.S.C. 7540",
        "10 U.S.C. 8612",
        "10 U.S.C. 9540",
    ]
    # Trailing number is the title of a following CFR citation, not a section.
    assert cites("41 U.S.C. 1303 and 48 CFR chapter 1") == ["41 U.S.C. 1303"]
    # A following USC citation starts a fresh match; "or" lists still expand.
    assert cites("10 U.S.C. 7317 and 17 U.S.C. 401 or 402") == [
        "10 U.S.C. 7317",
        "17 U.S.C. 401",
        "17 U.S.C. 402",
    ]
    # List then a separate chapter citation for the same title.
    assert cites("46 U.S.C. 12112 and 50501 and 46 U.S.C. chapter 551") == [
        "46 U.S.C. 12112",
        "46 U.S.C. 50501",
        "46 U.S.C. chapter 551",
    ]
    # Subsection parens and notes are kept in cite, stripped for section.
    assert sections("10 U.S.C. 2306a(b)(4)") == ["10 U.S.C. 2306a"]
    assert sections("10 U.S.C. 3204 note") == ["10 U.S.C. 3204"]
    # Prose after a comma is not consumed.
    assert cites("10 U.S.C. 4864, Miscellaneous Limitations on the Procurement") == [
        "10 U.S.C. 4864"
    ]
    # Prose form, periodless form, and ranges.
    assert cites("section 501(c)(3) of title 26") == ["26 U.S.C. 501(c)(3)"]
    assert sections("section 501(c)(3) of title 26") == ["26 U.S.C. 501"]
    assert cites("10 USC 2433(d)") == ["10 U.S.C. 2433(d)"]
    assert cites("eligible under 41 U.S.C. 8502-8504.") == ["41 U.S.C. 8502-8504"]
    print("selftest: all assertions passed")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(DATA_DIR.glob("title-48_*.xml"))
    for xml_path in tqdm(xml_files, desc="Extracting USC citations"):
        result = parse_file(xml_path)
        date = xml_path.stem.replace("title-48_", "")
        out_path = DATA_DIR / f"usc_citations_{date}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
