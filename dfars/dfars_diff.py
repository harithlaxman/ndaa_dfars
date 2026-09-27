"""Diff the DFARS nodes each NDAA section amended, before vs after.

For every NDAA in ``data/single_ndaa_frcases.csv`` we already know two things up
front, so no eCFR/Federal-Register call is needed at diff time:

  * the DFARS sections it amended — the ``sections`` column (the flattened
    ``get_sections().implements`` output produced by scrape_fr.py); and
  * the before/after DFARS snapshot dates — from the manifest
    ``data/dfars_diffs.csv`` that scrape_ecfr.py wrote (it resolved each case's
    snapshot pair against the eCFR versions API).

So the job reduces to: read the amended sections from the CSV, load the
before/after snapshots from Mongo (``ndaa_dfars.dfars``), and diff those
sections' node text. Output schema (consumed by the drafting pipeline) is
``{n_sections, sections:[{ndaa_year, ndaa_section, amended_sections, n_cases,
cases:[{case, before_snapshot, after_snapshot, before_present, after_present,
changes:[...]}]}]}``.
"""
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from utils.mongo_utils import getMongoClient, get_dfars_version

_ROOT = Path(__file__).resolve().parent.parent
FR_CASES = _ROOT / "data" / "single_ndaa_frcases.csv"
# scrape_ecfr.py's manifest: resolved before/after snapshot dates per case.
MANIFEST = _ROOT / "data" / "dfars_diffs.csv"
OUT_DIR = _ROOT / "data"
DB = "ndaa_dfars"
COLL = "dfars"

_WS = re.compile(r"\s+")

# The ``before`` of a clause/provision added in Part 252 (see _parent_before).
PART_252_BEFORE = "Part 252 - SOLICITATION PROVISIONS AND CONTRACT CLAUSES"


def parse_date(s: str) -> datetime:
    """Parse a manifest ISO date (YYYY-MM-DD) into a datetime."""
    return datetime.strptime(s.strip(), "%Y-%m-%d")


def part_of(section_number: str) -> str:
    return section_number.split(".")[0]


def _norm(text: str) -> str:
    """Whitespace-collapsed text, for the equality test only."""
    return _WS.sub(" ", text or "").strip()


def _heading(section: dict) -> str:
    """The node's heading, minus its leading number for SECTION/SUBSECTION.

    ``section.heading`` is the HEAD text, e.g. "201.104 Applicability." ->
    "Applicability."; PART/SUBPART headings ("PART 201—FEDERAL ...") are kept
    whole. Docs ingested before ``heading`` was stored fall back to the first
    line of ``section.text``, which is the same HEAD text.
    """
    head = (section.get("heading")
            or (section.get("text") or "").split("\n", 1)[0]).strip()
    number = section.get("number", "")
    if number and head.startswith(number):
        return head[len(number):].strip()
    return head


# ─── core diff ────────────────────────────────────────────────────────────────


def _in_scope(number: str, wanted: set[str]) -> bool:
    """True if ``number`` is a wanted section or a sub-node of one.

    A node is in scope when its section_number is one of the FR-amended numbers,
    or a descendant (e.g. listed "236.606" also pulls in subsection "236.606-70"),
    so a parent-level amendatory instruction still captures the child that changed.
    """
    return number in wanted or any(number.startswith(w + "-") for w in wanted)


def _parent_before(number: str, bfull: dict[str, dict],
                   afull: dict[str, dict]) -> tuple[str | None, str]:
    """``(ancestor_number, before_text)`` standing in for an added node's before.

    Walks up the added node's parent chain (as it sits in the after snapshot) to
    the nearest ancestor that exists in the before snapshot, skipping ancestors
    that are themselves new (e.g. ``225.7019`` when ``225.7019-1`` is added with
    it). The text is that ancestor as it read before the insertion: its own text
    followed by each existing direct child's own text, in document order, i.e. the
    siblings the new node joins. ``(None, "")`` if no ancestor existed.

    In Part 252 the parent is usually a synthesized section (e.g. ``252.225``)
    whose children are dozens of unrelated clauses, so an added clause instead
    gets just the part title, ``PART_252_BEFORE``, attributed to ``252``.
    """
    if part_of(number) == "252":
        return "252", PART_252_BEFORE
    cur = (afull.get(number) or {}).get("parent")
    while cur is not None and cur not in bfull:
        cur = (afull.get(cur) or {}).get("parent")
    if cur is None:
        return None, ""
    parent = bfull[cur]
    nodes = [parent] + [bfull[c] for c in parent.get("children", []) if c in bfull]
    texts = [(n.get("text") or "").strip() for n in nodes]
    return cur, "\n\n".join(t for t in texts if t)


def _diff_maps(bmap: dict[str, dict], amap: dict[str, dict],
               bfull: dict[str, dict] | None = None,
               afull: dict[str, dict] | None = None) -> list[dict]:
    """Changed nodes given {section_number: section-subdoc} maps for before/after.

    ``bfull``/``afull`` are the unscoped snapshots, used to give an added node its
    parent's before text (see ``_parent_before``); they default to the scoped maps.
    """
    bfull = bmap if bfull is None else bfull
    afull = amap if afull is None else afull
    changes = []
    for number in sorted(bmap.keys() | amap.keys()):
        bsec, asec = bmap.get(number), amap.get(number)
        before_text = (bsec or {}).get("text", "")
        after_text = (asec or {}).get("text", "")
        # Skip when there's no own-text change. The heading is part of the text,
        # so renames are caught here too. This also drops structural-only nodes
        # (e.g. a synthesized SECTION parent of -7xxx clauses) that appear or
        # vanish in the hierarchy graph with empty text on both sides.
        if _norm(before_text) == _norm(after_text):
            continue
        before_from = number if bsec else None
        if bsec and asec:
            status = "modified"
        elif asec:
            status = "added"
            # An added node has no before of its own; show the parent it was
            # inserted into (with its existing children), and record which node
            # that text came from.
            before_from, before_text = _parent_before(number, bfull, afull)
        else:
            status = "deleted"
        changes.append({
            "part": part_of(number),
            "number": number,
            "type": (asec or bsec).get("type"),
            "status": status,
            "heading": _heading(asec or bsec),
            "before": before_text,
            "before_from": before_from,
            "after": after_text,
        })
    return changes


def diff_sections(before: list[dict], after: list[dict],
                  numbers: set[str]) -> list[dict]:
    """Changed nodes among specific section_numbers (and their sub-nodes).

    Snapshots carry the whole DFARS, so scoping is purely by section_number and
    independent of part. A section that didn't change between the two snapshots
    simply yields no entry.
    """
    bfull = {d["section_number"]: d["section"] for d in before}
    afull = {d["section_number"]: d["section"] for d in after}
    bmap = {n: s for n, s in bfull.items() if _in_scope(n, numbers)}
    amap = {n: s for n, s in afull.items() if _in_scope(n, numbers)}
    return _diff_maps(bmap, amap, bfull, afull)


# ─── inputs ───────────────────────────────────────────────────────────────────


def load_sections_by_id() -> dict[str, dict]:
    """``{_id: {ndaa_year, ndaa_section, sections:set}}`` from single_ndaa_frcases.csv.

    ``_id`` is ``YEAR_SECTION``. The ``sections`` cell groups FR-amended DFARS
    section numbers by doc (``;``) and part (``,``); since diffing is by section
    number alone, we flatten it all into one set. Rows that repeat an ``_id``
    (a section implemented by two distinct rules) are merged.
    """
    out: dict[str, dict] = {}
    with FR_CASES.open() as f:
        for row in csv.DictReader(f):
            _id = (row.get("_id") or "").strip()
            if "_" not in _id:
                continue
            year, section = _id.split("_", 1)
            raw = (row.get("sections") or "").replace(";", ",")
            secs = {s.strip() for s in raw.split(",") if s.strip()}
            entry = out.setdefault(
                _id, {"ndaa_year": year, "ndaa_section": section, "sections": set()}
            )
            entry["sections"].update(secs)
    return out


def load_manifest() -> dict[tuple[str, str], list[dict]]:
    """``{(ndaa_year, ndaa_section): [{case, before, after}]}`` from the manifest.

    Each entry is one resolved before→after snapshot pair (a dfars_case).
    Exact-duplicate rows are dropped so a section isn't diffed twice for the
    same pair.
    """
    out: dict[tuple[str, str], list[dict]] = {}
    seen: set[tuple] = set()
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            year = (row.get("ndaa_year") or "").strip()
            section = (row.get("ndaa_section") or "").strip()
            before = (row.get("before_date") or "").strip()
            after = (row.get("after_date") or "").strip()
            case = (row.get("dfars_case") or "").strip()
            if not (year and section and before and after):
                continue
            dedup = (year, section, case, before, after)
            if dedup in seen:
                continue
            seen.add(dedup)
            out.setdefault((year, section), []).append({
                "case": case,
                "before": parse_date(before),
                "after": parse_date(after),
            })
    return out


# ─── orchestration ────────────────────────────────────────────────────────────


def _load_version(client, date: datetime, cache: dict) -> list[dict]:
    """Fetch a DFARS snapshot from Mongo, memoized by date."""
    if date not in cache:
        cache[date] = get_dfars_version(client, DB, COLL, date)
    return cache[date]


def diff_ndaa(client, info: dict, manifest: dict, cache: dict) -> dict:
    """Diff one NDAA: its amended sections across each before/after snapshot pair."""
    year, section = info["ndaa_year"], info["ndaa_section"]
    sections = info["sections"]
    cases = []
    for entry in manifest.get((year, section), []):
        before = _load_version(client, entry["before"], cache)
        after = _load_version(client, entry["after"], cache)
        cases.append({
            "case": entry["case"],
            "before_snapshot": entry["before"].date().isoformat(),
            "after_snapshot": entry["after"].date().isoformat(),
            "before_present": bool(before),
            "after_present": bool(after),
            "changes": diff_sections(before, after, sections),
        })
    return {
        "ndaa_year": year,
        "ndaa_section": section,
        "amended_sections": sorted(sections),
        "n_cases": len(cases),
        "cases": cases,
    }


def _print_section(report: dict, verbose: bool = True) -> None:
    year, section = report["ndaa_year"], report["ndaa_section"]
    total = sum(len(c["changes"]) for c in report["cases"])
    print(f"NDAA {year} sec {section}: {report['n_cases']} case(s), "
          f"{total} changed node(s)")
    if not report["cases"]:
        print(f"  WARNING: no snapshot pair in manifest for {year}_{section}; "
              f"no diff produced")
        return
    for c in report["cases"]:
        if not c["before_present"] or not c["after_present"]:
            miss = "before" if not c["before_present"] else "after"
            print(f"  case {c['case']}  WARNING: no {miss} snapshot in Mongo "
                  f"({c['before_snapshot']} -> {c['after_snapshot']})")
        print(f"  case {c['case']}  before {c['before_snapshot']} -> after "
              f"{c['after_snapshot']}  {len(c['changes'])} changed node(s)")
        if verbose:
            for ch in c["changes"]:
                print(f"      {ch['status']:>8}  {ch['number']:<16} ({ch['type']})")


def run_all() -> None:
    """Diff every NDAA in single_ndaa_frcases.csv into one combined file."""
    client = getMongoClient()
    cache: dict = {}
    by_id = load_sections_by_id()
    manifest = load_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for info in by_id.values():
        report = diff_ndaa(client, info, manifest, cache)
        reports.append(report)
        _print_section(report, verbose=False)

    combined = {"n_sections": len(reports), "sections": reports}
    out_path = OUT_DIR / "dfars_diff_all.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    total = sum(len(c["changes"]) for r in reports for c in r["cases"])
    print(f"\n{len(reports)} sections, {total} changed nodes total. Wrote {out_path}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python dfars/dfars_diff.py <ndaa_id>   e.g. 2024_2881\n"
                 "       python dfars/dfars_diff.py --all")
    if sys.argv[1] in ("--all", "all"):
        run_all()
        return

    _id = sys.argv[1]
    by_id = load_sections_by_id()
    if _id not in by_id:
        sys.exit(f"NDAA id {_id!r} not found in {FR_CASES.name}")

    manifest = load_manifest()
    client = getMongoClient()
    report = diff_ndaa(client, by_id[_id], manifest, {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"dfars_diff_{_id}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _print_section(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
