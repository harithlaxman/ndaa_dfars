"""Shared utilities for the framework2 pipeline."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

_FRAMEWORK_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FRAMEWORK_DIR.parents[2]   # ndaa_dfars/ -- holds data/
_DATA_DIR = _REPO_ROOT / "data"

FR_CASES_PATH = _DATA_DIR / "single_ndaa_frcases.csv"
DIFF_PATH = _DATA_DIR / "dfars_diff_all.json"
MANIFEST_PATH = _REPO_ROOT / "pipeline" / "out" / "manifests_fr_cases.json"


def _section_key(change: dict) -> str:
    """Roll a changed DFARS node up to its drafting unit.

    - part 252 clauses (e.g. 252.204-7012) stay whole -- they are self-contained
      provisions/clauses, not subsections of a prescriptive section.
    - a SECTION node (or any number without a '-' suffix) is its own unit.
    - a SUBSECTION (e.g. 236.606-70) rolls up to its enclosing SECTION (236.606).
    """
    number = change["number"]
    if change.get("part") == "252":
        return number
    if change.get("type") == "SECTION" or "-" not in number:
        return number
    return number.rsplit("-", 1)[0]


def get_single_ndaa_cases() -> set[str]:
    """Case numbers (e.g. '2024-D019') that implement exactly one NDAA section.

    Derived from data/fr_cases.csv, which carries one row per
    (NDAA section, FR case). A case appearing against only a single
    (ndaa_year, ndaa_section) pair implements exactly one NDAA section; cases
    spanning several NDAA sections are excluded. Cases absent from fr_cases.csv
    cannot be classified and are therefore not single.
    """
    case_to_ndaas: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with open(FR_CASES_PATH) as f:
        for row in csv.DictReader(f):
            case_to_ndaas[row["case_number"]].add(
                (str(row["ndaa_year"]), str(row["ndaa_section"]))
            )
    return {case for case, ndaas in case_to_ndaas.items() if len(ndaas) == 1}


def get_single_ndaa_allowed() -> dict[tuple[str, str], set[str]]:
    """Map (ndaa_year, ndaa_section) -> DFARS section units that come from cases
    implementing exactly one NDAA section.

    Used by both agent.py/baseline (to filter groups before running) and eval.py
    (to filter results after running), so both score the same single-NDAA-case
    population. Changed DFARS nodes are rolled up to their section unit via
    `_section_key`, matching the drafting units `_group_sections` produces.
    """
    single_cases = get_single_ndaa_cases()

    with open(DIFF_PATH) as f:
        diff = json.load(f)

    allowed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in diff.get("sections", []):
        key = (str(entry["ndaa_year"]), str(entry["ndaa_section"]))
        for case in entry.get("cases", []):
            if case.get("case") not in single_cases:
                continue
            for ch in case.get("changes", []):
                allowed[key].add(_section_key(ch))
    return dict(allowed)


def expected_section_units(
    single_ndaa: bool = True,
) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """The DFARS section units the pipeline was *expected* to draft per NDAA.

    Returns ``{(ndaa_year, ndaa_section): {section_unit: {"before", "after"}}}``.

    Mirrors ``agent.load_ndaa_groups`` selection exactly -- manifest required,
    changes pooled across the NDAA's cases, rolled up to section units via
    ``_section_key`` with before/after concatenated in node-number order, pure
    additions (no prior text) skipped, single-NDAA allowed filter, groups with
    >25 units skipped -- but needs no Mongo (the NDAA statutory text that
    ``load_ndaa_groups`` fetches isn't needed to know *which* sections were due).

    Used by eval.py to penalize required sections a run failed to emit: anything
    in here but absent from a run's ``section_drafts`` is a (recall) miss.
    """
    with open(DIFF_PATH) as f:
        diff = json.load(f)
    with open(MANIFEST_PATH) as f:
        manifest_ids = {s["ndaa_id"] for s in json.load(f).get("sections", [])}

    allowed = get_single_ndaa_allowed() if single_ndaa else None

    expected: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for entry in diff.get("sections", []):
        year = str(entry["ndaa_year"])
        section = str(entry["ndaa_section"])
        if f"{year}_{section}" not in manifest_ids:
            continue

        changes: list[dict] = []
        for case in entry.get("cases", []):
            changes.extend(case.get("changes", []))
        if not changes:
            continue

        by_key: dict[str, list[dict]] = defaultdict(list)
        for ch in changes:
            by_key[_section_key(ch)].append(ch)

        secs: dict[str, dict[str, str]] = {}
        for key, nodes in by_key.items():
            nodes = sorted(nodes, key=lambda c: c["number"])
            before = "\n\n".join(
                n["before"].strip() for n in nodes if n.get("before", "").strip()
            )
            after = "\n\n".join(
                n["after"].strip() for n in nodes if n.get("after", "").strip()
            )
            if not before:  # pure addition -- nothing to draft "from"; skip
                continue
            secs[key] = {"before": before, "after": after}

        if allowed is not None:
            allowed_secs = allowed.get((year, section))
            if not allowed_secs:
                continue
            secs = {
                k: v for k, v in secs.items()
                if any(a in k for a in allowed_secs)
            }

        if not secs or len(secs) > 25:
            continue
        expected[(year, section)] = secs

    return expected
