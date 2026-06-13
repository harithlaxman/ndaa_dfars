"""Shared utilities for the framework2 pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

_FRAMEWORK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FRAMEWORK_DIR.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

DOC_TO_NDAA_PATH = _DATA_DIR / "doc_to_ndaa.csv"
VALID_PAIRS_PATH = _DATA_DIR / "valid_pairs.json"


def get_single_ndaa_allowed() -> dict[tuple[str, str], set[str]]:
    """Return (ndaa_year, ndaa_section) -> set of DFARS section names that
    come from document numbers with exactly one NDAA citation.

    Used by both agent.py (to filter groups before running) and eval.py
    (to filter results after running).
    """
    # Identify document_numbers with exactly one NDAA citation
    single_docs: set[str] = set()
    with open(DOC_TO_NDAA_PATH) as f:
        for row in csv.DictReader(f):
            if len(row["ndaa_citations"].split(";")) == 1:
                single_docs.add(row["document_number"])

    # Map (ndaa_year, ndaa_section) -> set of DFARS section names
    # from single-NDAA document numbers
    with open(VALID_PAIRS_PATH) as f:
        pairs = json.load(f)

    allowed: dict[tuple[str, str], set[str]] = {}
    for p in pairs:
        if p["dfars"]["document_number"] not in single_docs:
            continue
        dfars_sec = p["dfars"].get("section")
        if not dfars_sec:
            continue
        for ndaa in p.get("ndaas", []):
            key = (str(ndaa["year"]), str(ndaa["section"]))
            allowed.setdefault(key, set()).add(dfars_sec)

    return allowed
