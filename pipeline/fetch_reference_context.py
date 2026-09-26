#!/usr/bin/env python3
"""Resolve one NDAA section's external citations into short, plain-English explanations.

NDAA sections are dense statutory text that rarely stand on their own: they amend a U.S. Code
section, point back to an earlier NDAA, or lean on a public law / Statutes-at-Large citation.
A downstream drafter implementing the section inside DFARS struggles while those cross-references
are still unresolved.

This stage is a lighter sibling of ``fetch_drafting_context.py``. It runs the same research loop
over the GovInfo text fetchers and the "other NDAA section" lookup to pull in the external sources
the section references, but its output is deliberately narrow: just the RESOLVED DEPENDENCIES — for
each external citation the section relies on, a short explanation of what that authority says and
what the NDAA amendment actually changes about it. It produces no overview and no classified
requirement list, and it does NOT decide where the change belongs in DFARS or how to draft it.

Reads the section from Mongo (``ndaa_dfars.ndaas``); writes the result as JSON to
``pipeline/out/``. Never writes to Mongo.

Run as a script, it processes every NDAA section the DFARS diff covers: the batch's
(year, section) list is sourced from ``data/dfars_diff_all.json`` (the same file
framework2/framework1 consume), so coverage lines up with the drafting pipeline.

Usage:
    uv run python pipeline/fetch_reference_context.py
"""

import copy
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Reuse the research tooling, prompt builder, and CSV helpers from the manifest stage rather
# than duplicating ~200 lines of tool schemas. Both files live in pipeline/, so the script's
# own directory is already on sys.path; add it explicitly to be safe.
_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import fetch_context as fc  # noqa: E402

from utils.mongo_utils import get_doc_by_year_section  # noqa: E402
from utils.openai import (  # noqa: E402
    connect_to_openai,
    get_structured_response_from_input,
    run_tool_loop,
)

OUT_DIR = fc.OUT_DIR
DATA_DIR = fc.DATA_DIR
DIFF_FILE = DATA_DIR / "dfars_diff_all.json"
DB = fc.DB
NDAAS = fc.NDAAS


# ─── Resolved-dependency schema ───────────────────────────────────────────────────────

class ResolvedDependency(BaseModel):
    """One external citation the section relies on, resolved and explained briefly."""

    reference: str = Field(
        description=(
            "The external citation as it appears in (or is implied by) the NDAA section, "
            'e.g. "10 U.S.C. 2304", "section 847 of the NDAA for FY2017", '
            '"Public Law 111-78", or "124 Stat 2859".'
        )
    )
    explanation: str = Field(
        description=(
            "Short plain-English explanation of what this referenced authority says and what "
            "the NDAA section's amendment actually changes about it. Keep it to the essence — "
            "the concrete effect of the change on the cited source — not a restatement of the "
            "statute. If the reference could not be resolved, state the amendatory instruction "
            "as far as the NDAA's own words support it and note that it is unresolved."
        )
    )


class SectionReferences(BaseModel):
    ndaa_id: str
    fiscal_year: int
    section_number: str
    section_heading: str
    dependencies: list[ResolvedDependency] = Field(
        default_factory=list,
        description=(
            "The external dependencies the section relies on, one per entry, each resolved into "
            "a short explanation of what the amendment actually changed."
        ),
    )


# ─── Tools (external fetchers only) ───────────────────────────────────────────────────

# NDAA fiscal year of the section currently being processed, set per section in
# process_section. The get_usc_section wrapper uses it to pick the U.S. Code edition.
_ndaa_year: int | None = None


def _usc_with_year_fallback(title, section, usc_type="usc", year=None):
    """Fetch a U.S. Code section at the edition just before this NDAA amended it.

    Deterministic and model-agnostic: ignores any model-supplied ``year``. Tries the
    (NDAA fiscal year - 1) edition first -- the one that does not yet contain this NDAA's
    change -- then, if that edition has no text (GovInfo lacks it), the NDAA fiscal year
    edition. Returns the first edition that has text, or None if neither does.
    """
    if _ndaa_year is None:
        return fc.get_usc_section(title, section, usc_type=usc_type, year=year)
    for y in (_ndaa_year - 1, _ndaa_year):
        text = fc.get_usc_section(title, section, usc_type=usc_type, year=y)
        if text:
            return text
    return None


# Reuse the manifest stage's research tools, minus the DFARS semantic search: this stage only
# resolves *external* sources — deciding where/how the change lands in DFARS is the downstream
# drafting agent's job. get_usc_section's edition is chosen deterministically by the wrapper
# above, so its schema drops the model-facing `year` argument.
TOOLS = []
for _t in fc.TOOLS:
    if _t["name"] == "get_dfars_context":
        continue
    if _t["name"] == "get_usc_section":
        _t = copy.deepcopy(_t)
        _t["parameters"]["properties"].pop("year", None)
        _t["description"] = (
            "Fetch the text of a United States Code section (e.g. title 10, section 2304) as it "
            "stood just before this NDAA amended it. Do not pass a year; the edition is selected "
            "automatically."
        )
    TOOLS.append(_t)

DISPATCH = {k: v for k, v in fc.DISPATCH.items() if k != "get_dfars_context"}
DISPATCH["get_usc_section"] = _usc_with_year_fallback


# ─── Prompt + driver ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal research analyst for U.S. Defense Acquisition Regulation.

You are given the text of one section of a National Defense Authorization Act (NDAA). Downstream,
a drafting agent will implement this section inside the Defense Federal Acquisition Regulation
Supplement (DFARS). The NDAA text often contains references to external documents like U.S. Code,
Public Law, Statutes at Large, other NDAAs, or even other sections of the same NDAA. This makes it
difficult for that drafter to understand the full requirement without resolving these indirect
references.

Your job is narrow: resolve those external references and, for each one, explain in a few words
what the referenced authority says and what this NDAA section actually changes about it. You do NOT
write an overview, you do NOT classify or enumerate the section's requirements, and you do NOT
decide where the change belongs in DFARS or how to draft it.

Work in two phases:

1. RESEARCH. If the section cannot be understood on its own, call the
   provided tools to resolve any external reference you need:
   - When the section amends or references a U.S. Code section, fetch it (get_usc_section(title, section)) so you
     understand what the text being amended actually says. Always prefer calling the tool with
     the section number without any suffixes  (e.g., ``2403`` and not ``2403-1``). Do not pass a year — the
     correct edition (the one just before this NDAA amended the Code) is selected for you automatically.
   - When it references another NDAA (commonly "section NNN of the National Defense Authorization
     Act for Fiscal Year YYYY"), fetch that section (get_ndaa_section, year=YYYY, section=NNN).
   - When it references a public law or a Statutes at Large citation, fetch it.
   Fan out to as many tool calls as you need, but only if its needed to understand the NDAA section completely.
   If a tool call fails or comes back empty, retry it once with corrected arguments when the
   problem looks like a bad argument. If it still fails — or every tool call fails — do not abandon
   the entry: fall back to what the NDAA text itself states and flag the gap (see below).

2. SYNTHESIZE. Turn what you gathered into a list of resolved dependencies — one per external
   reference the section relies on:
   - reference: the external citation as it appears in (or is implied by) the section.
   - explanation: a short, plain-English explanation of what that authority says and what this
     NDAA section's amendment actually changes about it. Capture the concrete effect of the change
     on the cited source; do not restate the statute line by line.

When you could NOT resolve a reference (a tool failed, or every tool call failed), still include
the dependency:
   - reference: the citation as the NDAA states it.
   - explanation: state the amendatory instruction as far as the NDAA's own words support it
     (e.g. "Amends <cited section> by striking X and inserting Y"), and note that the reference
     could not be resolved. Do NOT invent the contents of the unresolved reference, and do NOT
     drop the dependency.

After researching, output the resolved dependencies in the required structured format."""


def process_section(llm, doc: dict) -> SectionReferences:
    """Run the research-then-synthesize loop for one NDAA section document."""
    # Pin the NDAA fiscal year so the get_usc_section wrapper fetches the edition
    # in effect just before this NDAA (NDAA year - 1).
    global _ndaa_year
    _ndaa_year = int(doc["fiscal_year"])

    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fc.build_user_prompt(doc)},
    ]
    conversation = run_tool_loop(llm, input_messages, TOOLS, DISPATCH)
    conversation.append(
        {
            "role": "user",
            "content": (
                "Now output the resolved dependencies in the required structured format, using "
                "the context you gathered: one entry per external reference, each with the "
                "citation and a short explanation of what the amendment actually changed. If any "
                "reference could not be resolved, still include it from the NDAA text and note "
                "that it is unresolved."
            ),
        }
    )
    result = get_structured_response_from_input(llm, conversation, SectionReferences)

    # Pin the identity fields from the source doc rather than trusting the model.
    section = doc["section"]
    result.ndaa_id = doc["_id"]
    result.fiscal_year = doc["fiscal_year"]
    result.section_number = str(section["number"])
    result.section_heading = section["heading"]
    return result


def run_one(year: str, section: str) -> None:
    """Process a single section and write its own reference-context JSON file."""
    doc = get_doc_by_year_section(fc._client(), DB, NDAAS, int(year), section)
    if not doc:
        print(f"No NDAA section found for {year}_{section}")
        sys.exit(1)

    print(f"Processing NDAA {year}_{section}: {doc['section'].get('heading', '')}")
    result = process_section(connect_to_openai(), doc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"references_{year}_{section}.json"
    out_path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"Wrote {out_path} — {len(result.dependencies)} dependencies")


def diff_sections(diff_path: Path) -> list[tuple[str, str]]:
    """Read the DFARS diff; return de-duplicated (ndaa_year, ndaa_section) pairs in file order.

    These are exactly the NDAA sections framework1/framework2 look context up for (keyed by
    "<year>_<section>"), so sourcing pairs here keeps coverage aligned with the drafting pipeline.
    """
    diff = json.loads(diff_path.read_text())
    pairs: list[tuple[str, str]] = []
    seen = set()
    for entry in diff.get("sections", []):
        year = str(entry.get("ndaa_year", "")).strip()
        section = str(entry.get("ndaa_section", "")).strip()
        if not year or not section:
            continue
        key = (year, section)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def run_pairs(pairs: list[tuple[str, str]], out_name: str) -> None:
    """Process a list of (year, section) NDAA sections into one combined JSON file."""
    print(f"Processing {len(pairs)} unique NDAA sections")
    llm = connect_to_openai()
    client = fc._client()

    sections, not_found, failed = [], [], []
    for i, (year, section) in enumerate(pairs, 1):
        tag = f"{year}_{section}"
        doc = get_doc_by_year_section(client, DB, NDAAS, int(year), section)
        if not doc:
            print(f"[{i}/{len(pairs)}] {tag}: not found in Mongo, skipping")
            not_found.append(tag)
            continue
        print(f"[{i}/{len(pairs)}] {tag}: {doc['section'].get('heading', '')}")
        try:
            result = process_section(llm, doc)
            sections.append(result.model_dump())
        except Exception as e:  # keep the batch going if one section fails
            print(f"  FAILED {tag}: {e}")
            failed.append(tag)

    report = {
        "n_sections": len(sections),
        "n_not_found": len(not_found),
        "n_failed": len(failed),
        "not_found": not_found,
        "failed": failed,
        "sections": sections,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"Wrote {out_path} — {len(sections)} sections, "
        f"{len(not_found)} not found, {len(failed)} failed"
    )


def run_csv(csv_path: Path, out_name: str) -> None:
    """Process every unique NDAA section in a CSV into one combined JSON file."""
    run_pairs(fc.csv_sections(csv_path), out_name)


if __name__ == "__main__":
    run_pairs(diff_sections(DIFF_FILE), "reference_context_fr_cases.json")
    # run_one("2024", "865")
