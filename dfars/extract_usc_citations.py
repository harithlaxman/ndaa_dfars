"""Regex-extract U.S. Code citations and attach them to each DFARS docs node.

For every `data/DFARS/docs/title-48_<date>.json` file, scan each node's own `text`
and write the distinct USC citations found back onto that node, in place, as a
`usc_citations` list:

    "225.7002": {
      ...,
      "usc_citations": [
        {"cite": "10 U.S.C. 4862(k)", "section": "10 U.S.C. 4862", "raw": "10 U.S.C. 4862(k)"},
        ...
      ]
    }

Every node gets the field (an empty list when it cites nothing), so the operation
is idempotent — re-running overwrites the list rather than appending.

Within a citation entry, `cite` is the citation as written (subsection parens and
`note` kept), `section` is normalized to the bare section level (parens/`note`
stripped) so it lines up with the section-level USC citations the NDAA extractor
produces; `raw` is the full matched span the item came from.

The extractor handles the list forms that appear in the data, e.g.
`10 U.S.C. 7504, 8354 and 3253` expands to three separate citations, while
refusing to swallow the title of a *following* citation: in
`41 U.S.C. 1303 and 48 CFR chapter 1` only `41 U.S.C. 1303` is captured.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = _PROJECT_ROOT / "data" / "DFARS" / "docs"

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


# One edit in an amendment history: a Federal Register citation followed by a date,
# e.g. "80 FR 51745, Aug. 26, 2015". Extra pinpoint pages ("76 FR 6006, 6008, ...")
# are ignored; the cite keeps the volume and starting page. The separator before
# the date tolerates the comma/period/stray-">" forms that appear in the data.
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_EDIT_RE = re.compile(
    r"(?P<cite>\d+\s+FR\s+\d+)(?:,\s*\d+)*"
    r"[,.]?\s*>?\s*"
    r"(?P<month>[A-Za-z]+)\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
)


def parse_amendment_history(raw: str) -> list[dict]:
    """Parse a raw CITA string into an ordered list of {cite, date} edits.

    The first edit is the original publication; the rest are amendments (and
    redesignations) in the order they appear. `cite` is the "<vol> FR <page>"
    citation; `date` is a datetime. Items without a recognizable date are skipped.
    """
    edits = []
    for m in _EDIT_RE.finditer(raw):
        month = _MONTHS.get(m.group("month").lower())
        if month is None:
            continue
        edits.append({
            "cite": re.sub(r"\s+", " ", m.group("cite")),
            "date": datetime(int(m.group("year")), month, int(m.group("day"))),
        })
    return edits


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

    # Amendment-history parsing: original + amendments, in order, as datetimes.
    def hist(raw: str) -> list[tuple[str, str]]:
        return [(e["cite"], e["date"].strftime("%Y-%m-%d")) for e in parse_amendment_history(raw)]

    assert hist("[80 FR 51745, Aug. 26, 2015, as amended at 80 FR 56930, Sept. 21, 2015]") == [
        ("80 FR 51745", "2015-08-26"),
        ("80 FR 56930", "2015-09-21"),
    ]
    # Pinpoint page kept out of the cite; "June"/"July" long forms.
    assert hist("[76 FR 6006, 6008, Feb. 2, 2011, as amended at 74 FR 37647, July 29, 2009]") == [
        ("76 FR 6006", "2011-02-02"),
        ("74 FR 37647", "2009-07-29"),
    ]
    # Stray ">", a "." separator, and a "Redesignated at" connector all tolerated.
    assert hist("[76 FR 52142, >Aug. 19, 2011]") == [("76 FR 52142", "2011-08-19")]
    assert hist("[65 FR 14401. Mar. 16, 2000]") == [("65 FR 14401", "2000-03-16")]
    assert hist("[65 FR 50144, Aug. 17, 2000. Redesignated at 75 FR 51417, Aug. 20, 2010]") == [
        ("65 FR 50144", "2000-08-17"),
        ("75 FR 51417", "2010-08-20"),
    ]
    print("selftest: all assertions passed")


def main() -> None:
    """Annotate every docs node with its `usc_citations`, in place."""
    doc_files = sorted(DOCS_DIR.glob("title-48_*.json"))
    total = 0
    for path in tqdm(doc_files, desc="Extracting USC citations"):
        nodes = json.loads(path.read_text())
        for node in nodes.values():
            cites = extract_usc_citations(node.get("text", ""))
            node["usc_citations"] = cites
            total += len(cites)
        path.write_text(json.dumps(nodes, indent=2, ensure_ascii=False))
    print(f"Processed {len(doc_files)} files, added {total} USC citations.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
