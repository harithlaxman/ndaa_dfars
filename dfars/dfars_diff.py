"""Before/after DFARS text for every node changed by an NDAA section's rule(s).

Driven by data/fr_cases.csv: each row is a DFARS rulemaking implementing an NDAA
section, with an effective_on date and the parts it amended (cfr_references). For a
given NDAA (year, section) we, per implementing case:

  1. read the snapshot pair dfars/scrape_ecfr.py already resolved for this case from its
     manifest (data/DFARS/dfars_diffs.csv): before_date/after_date, which it picked by
     querying the eCFR versions API for the date the amendment actually lands in eCFR
     (which can lag effective_on). Both are literal version_date keys in Mongo
     `ndaa_dfars.dfars`, so no eCFR call is needed here.
  2. ask the Federal Register full text which CFR sections THIS rule amended
     (utils.api_utils.get_sections, by citation + publication_date), then index
     those nodes (and their sub-nodes) in both snapshots, keyed on section_number,
     and compare each node's OWN text (section.text — subsections are separate
     sibling docs, never folded into the parent; see utils/get_dfars.py), so the
     comparison is exact node-by-node. If the FR sections can't be resolved we fall
     back to comparing every node in the affected parts (cfr_references).
  3. emit one record per CHANGED node with its before and after text:
       added    -> before "", after = new text
       deleted  -> before = old text, after ""
       modified -> both present and differing
     unchanged nodes are omitted.

Output: data/dfars_diff_<year>_<section>.json + a printed one-line summary.

Caveats:
  - Scoping to the FR rule's own amended sections isolates a single rule even when a
    co-effective rule touches the same parts on the same date. The whole-part
    fallback does not: a dated eCFR snapshot reflects EVERY change effective that
    date in those parts, so a co-effective rule's edits can appear too.
  - Snapshot dates come from the scrape manifest; a case absent from it (or one
    scrape_ecfr.py couldn't resolve) yields no snapshots (before/after null, empty
    changes). Re-run dfars/scrape_ecfr.py to refresh the manifest.

Usage:
    python dfars/dfars_diff.py 2024 2881
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from utils.api_utils import get_sections
from utils.mongo_utils import getMongoClient, get_dfars_version

_ROOT = Path(__file__).resolve().parent.parent
FR_CASES = _ROOT / "data" / "fr_cases.csv"
# Manifest scrape_ecfr.py writes when fetching snapshots; carries the resolved
# before/after snapshot dates per case, so we can reuse them without re-querying eCFR.
MANIFEST = _ROOT / "data" / "DFARS" / "dfars_diffs.csv"
OUT_DIR = _ROOT / "data"
DB = "ndaa_dfars"
COLL = "dfars"

_WS = re.compile(r"\s+")


def parse_date(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {s!r}")


def part_of(section_number: str) -> str:
    return section_number.split(".")[0]


def _norm(text: str) -> str:
    """Whitespace-collapsed text, for the equality test only."""
    return _WS.sub(" ", text or "").strip()


def _heading(section: dict) -> str:
    """Best-effort heading for a node.

    The schema no longer carries a separate `section.heading`; the heading is
    the first line of `section.text`, e.g. "201.104 Applicability.",
    "Subpart 201.1 - Purpose, Authority, Issuance", or
    "PART 201 - FEDERAL ACQUISITION REGULATIONS SYSTEM". Strip the leading
    number for SECTION/SUBSECTION, or the "<label> - " prefix for PART/SUBPART.
    """
    first = (section.get("text") or "").split("\n", 1)[0].strip()
    number = section.get("number", "")
    if number and first.startswith(number):
        return first[len(number):].strip()
    if " - " in first:
        return first.split(" - ", 1)[1].strip()
    return first


def _row_cases(row: dict) -> list[dict]:
    """Split one CSV row into per-case records: {case, citation, pub_date, effective, parts}.

    Rows bundling several cases (';' in publication_date) are split so each case
    gets its own citation, publication date, effective date, and cfr_references segment.
    """
    dates = [d for d in (row["publication_date"] or "").split(";") if d.strip()]
    eff_dates = [d for d in (row["effective_on"] or "").split(";") if d.strip()]
    ref_groups = (row["cfr_references"] or "").split(";")
    cases = [c.strip() for c in (row["dfars_case"] or "").split(";")]
    cites = [c.strip() for c in (row["citation"] or "").split(";")]
    out = []
    for i, d in enumerate(dates):
        refs = ref_groups[i] if i < len(ref_groups) else (ref_groups[-1] if ref_groups else "")
        parts = sorted({p.strip() for p in refs.split(",") if p.strip()}, key=int)
        eff_raw = eff_dates[i] if i < len(eff_dates) else (eff_dates[-1] if eff_dates else "")
        out.append({
            "case": cases[i] if i < len(cases) else (cases[0] if cases else ""),
            "citation": cites[i] if i < len(cites) else (cites[0] if cites else ""),
            "pub_date": parse_date(d),
            "effective": parse_date(eff_raw) if eff_raw else None,
            "parts": parts,
        })
    return out


def cases_for(ndaa_year: str, ndaa_section: str) -> list[dict]:
    """All per-case records for one NDAA (year, section), across every CSV row."""
    out = []
    with FR_CASES.open() as f:
        for row in csv.DictReader(f):
            if (row.get("ndaa_year") or "").strip() != str(ndaa_year):
                continue
            if (row.get("ndaa_section") or "").strip() != str(ndaa_section):
                continue
            for rec in _row_cases(row):
                rec["ndaa_year"] = str(ndaa_year)
                rec["ndaa_section"] = str(ndaa_section)
                out.append(rec)
    return out


def all_sections() -> list[tuple[str, str]]:
    """Unique (ndaa_year, ndaa_section) pairs in fr_cases.csv, in file order."""
    seen, out = set(), []
    with FR_CASES.open() as f:
        for row in csv.DictReader(f):
            year = (row.get("ndaa_year") or "").strip()
            section = (row.get("ndaa_section") or "").strip()
            if not year or not section or (year, section) in seen:
                continue
            seen.add((year, section))
            out.append((year, section))
    return out


def _node_map(version: list[dict], part: str) -> dict[str, dict]:
    """{section_number: section-subdoc} for all nodes in `part` of one snapshot."""
    return {
        d["section_number"]: d["section"]
        for d in version
        if part_of(d["section_number"]) == part
    }


def _diff_maps(bmap: dict[str, dict], amap: dict[str, dict]) -> list[dict]:
    """Changed nodes given {section_number: section-subdoc} maps for before/after."""
    changes = []
    for number in sorted(bmap.keys() | amap.keys()):
        bsec, asec = bmap.get(number), amap.get(number)
        before_text = (bsec or {}).get("text", "") if bsec else ""
        after_text = (asec or {}).get("text", "") if asec else ""
        # Skip when there's no own-text change. The heading is part of the text,
        # so renames are caught here too. This also drops structural-only nodes
        # (e.g. a synthesized SECTION parent of -7xxx clauses) that appear or
        # vanish in the hierarchy graph with empty text on both sides.
        if _norm(before_text) == _norm(after_text):
            continue
        if bsec and asec:
            status = "modified"
        elif asec:
            status = "added"
        else:
            status = "deleted"
        changes.append({
            "part": part_of(number),
            "number": number,
            "type": (asec or bsec).get("type"),
            "status": status,
            "heading": _heading(asec or bsec),
            "before": before_text,
            "after": after_text,
        })
    return changes


def diff_part(before: list[dict], after: list[dict], part: str) -> list[dict]:
    """Changed nodes in one part, each with before/after own-text."""
    return _diff_maps(_node_map(before, part), _node_map(after, part))


def _in_scope(number: str, wanted: set[str]) -> bool:
    """True if `number` is a wanted section or a sub-node of one.

    A node is in scope when its section_number is one of the FR-amended numbers,
    or a descendant (e.g. listed "236.606" also pulls in subsection "236.606-70"),
    so a parent-level amendatory instruction still captures the child that changed.
    """
    return number in wanted or any(number.startswith(w + "-") for w in wanted)


def diff_sections(before: list[dict], after: list[dict],
                  numbers: set[str]) -> list[dict]:
    """Changed nodes among specific section_numbers (and their sub-nodes).

    `numbers` are the sections an FR rule's amendatory instructions list (from
    utils.api_utils.get_sections); snapshots carry the whole DFARS, so scoping is
    purely by section_number and independent of part.
    """
    bmap = {d["section_number"]: d["section"]
            for d in before if _in_scope(d["section_number"], numbers)}
    amap = {d["section_number"]: d["section"]
            for d in after if _in_scope(d["section_number"], numbers)}
    return _diff_maps(bmap, amap)


def _load_version(client, date: datetime, cache: dict | None) -> list[dict]:
    """Fetch a snapshot, memoized by date when a cache dict is supplied."""
    if cache is None:
        return get_dfars_version(client, DB, COLL, date)
    if date not in cache:
        cache[date] = get_dfars_version(client, DB, COLL, date)
    return cache[date]


_manifest_dates: dict[tuple[str, str, str], tuple[datetime | None, datetime | None]] | None = None


def _manifest_lookup() -> dict[tuple[str, str, str], tuple[datetime | None, datetime | None]]:
    """{(ndaa_year, ndaa_section, case): (before_date, after_date)} from the scrape manifest.

    scrape_ecfr.py already resolved each case's snapshot pair (querying the eCFR versions
    API to find the date its amendment actually lands, which can lag effective_on) and
    recorded before_date/after_date here, so we reuse those rather than re-querying eCFR.
    """
    global _manifest_dates
    if _manifest_dates is None:
        _manifest_dates = {}
        with MANIFEST.open() as f:
            for row in csv.DictReader(f):
                key = (
                    (row.get("ndaa_year") or "").strip(),
                    (row.get("ndaa_section") or "").strip(),
                    (row.get("dfars_case") or "").strip(),
                )
                b = (row.get("before_date") or "").strip()
                a = (row.get("after_date") or "").strip()
                _manifest_dates[key] = (
                    parse_date(b) if b else None,
                    parse_date(a) if a else None,
                )
    return _manifest_dates


_amended_cache: dict[tuple[str, str], set[str] | None] = {}


def _amended_sections(citation: str, pub_date: str) -> set[str] | None:
    """FR-amended section_numbers for a case, via utils.api_utils.get_sections.

    Cached per (citation, pub_date) since one citation can back several cases.
    Returns None when the rule's sections can't be resolved (no citation, or an
    API/HTML failure), signaling the caller to fall back to whole-part diffing.
    """
    key = (citation, pub_date)
    if key not in _amended_cache:
        secs = get_sections(citation, pub_date) if citation else None
        _amended_cache[key] = (
            {s["section"].strip() for s in secs if s.get("section")}
            if secs else None
        )
    return _amended_cache[key]


def diff_case(client, case_rec: dict, cache: dict | None = None) -> dict:
    key = (case_rec["ndaa_year"], case_rec["ndaa_section"], case_rec["case"])
    before_date, after_date = _manifest_lookup().get(key, (None, None))

    before = _load_version(client, before_date, cache) if before_date else []
    after = _load_version(client, after_date, cache) if after_date else []

    # Scope to the sections THIS rule actually amended (per the FR full text);
    # fall back to every node in the affected parts if they can't be resolved.
    amended = _amended_sections(
        case_rec.get("citation", ""), case_rec["pub_date"].strftime("%m/%d/%Y")
    )
    if amended:
        changes = diff_sections(before, after, amended)
        scope = "sections"
    else:
        changes = []
        for part in case_rec["parts"]:
            changes.extend(diff_part(before, after, part))
        scope = "parts"

    return {
        "case": case_rec["case"],
        "citation": case_rec.get("citation", ""),
        "pub_date": case_rec["pub_date"].date().isoformat(),
        "effective_on": case_rec["effective"].date().isoformat() if case_rec["effective"] else None,
        "before_snapshot": before_date.date().isoformat() if before_date else None,
        "after_snapshot": after_date.date().isoformat() if after_date else None,
        "before_present": bool(before),
        "after_present": bool(after),
        "scope": scope,
        "parts": case_rec["parts"],
        "amended_sections": sorted(amended) if amended else None,
        "changes": changes,
    }


def diff_section(ndaa_year: str, ndaa_section: str, client=None,
                 cache: dict | None = None) -> dict:
    client = client or getMongoClient()
    cases = cases_for(ndaa_year, ndaa_section)
    return {
        "ndaa_year": str(ndaa_year),
        "ndaa_section": str(ndaa_section),
        "n_cases": len(cases),
        "cases": [diff_case(client, c, cache) for c in cases],
    }


def _print_section(report: dict, verbose: bool = True) -> None:
    year, section = report["ndaa_year"], report["ndaa_section"]
    total = sum(len(c["changes"]) for c in report["cases"])
    print(f"NDAA {year} sec {section}: {report['n_cases']} case(s), "
          f"{total} changed node(s)")
    for c in report["cases"]:
        if not c["before_present"] or not c["after_present"]:
            miss = "before" if not c["before_present"] else "after"
            print(f"  case {c['case']}  ({c['after_snapshot']})  "
                  f"WARNING: no {miss} snapshot in Mongo")
        if c.get("scope") == "sections":
            scope_str = f"sections={len(c['amended_sections'] or [])}"
        else:
            scope_str = f"parts={','.join(c['parts']) or '-'} (fallback)"
        print(f"  case {c['case']}  before {c['before_snapshot']} -> after "
              f"{c['after_snapshot']}  {scope_str}  "
              f"{len(c['changes'])} changed node(s)")
        if verbose:
            for ch in c["changes"]:
                print(f"      {ch['status']:>8}  {ch['number']:<16} ({ch['type']})")


def run_all() -> None:
    """Diff every NDAA (year, section) in fr_cases.csv into one combined file."""
    client = getMongoClient()
    cache: dict = {}
    sections = all_sections()
    reports = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year, section in sections:
        report = diff_section(year, section, client=client, cache=cache)
        reports.append(report)
        _print_section(report, verbose=False)

    combined = {"n_sections": len(reports), "sections": reports}
    out_path = OUT_DIR / "dfars_diff_all.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    total = sum(len(c["changes"]) for r in reports for c in r["cases"])
    print(f"\n{len(reports)} sections, {total} changed nodes total. Wrote {out_path}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("--all", "all"):
        run_all()
        return
    if len(sys.argv) != 3:
        sys.exit("usage: python dfars/dfars_diff.py <ndaa_year> <ndaa_section>\n"
                 "       python dfars/dfars_diff.py --all")
    year, section = sys.argv[1], sys.argv[2]
    report = diff_section(year, section)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"dfars_diff_{year}_{section}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    _print_section(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
