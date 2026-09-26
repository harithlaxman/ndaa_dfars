#!/usr/bin/env python3
"""Brief a DFARS drafter on one NDAA section by resolving its external context.

NDAA sections are dense statutory text that rarely stand on their own: they amend a U.S. Code
section, point back to an earlier NDAA, or lean on a public law / Statutes-at-Large citation.
A downstream agent then has to draft the DFARS implementation of the section — but it can't do
that well while those cross-references are still unresolved and the requirement is buried in
legalese.

This stage closes that gap. It runs a research loop over the GovInfo text fetchers and an
"other NDAA section" lookup to pull in exactly the external sources the section references, then
distills the result into a small SECTION BRIEF: a plain-English overview of what the section
requires (the logic, stripped of statutory language, with the resolved cross-references folded in)
and the discrete requirements it imposes (each tagged as an addition, modification, or deletion).
It deliberately stops there — it does NOT decide where the change belongs in DFARS or how to draft
it; that is the downstream drafting agent's job.

Reads the section from Mongo (``ndaa_dfars.ndaas``); writes the brief as JSON to
``pipeline/out/``. Never writes to Mongo.

Run as a script, it briefs every NDAA section the DFARS diff covers: the batch's
(year, section) list is sourced from ``data/dfars_diff_all.json`` (the same file
framework2 consumes), so brief coverage lines up with what the drafting pipeline
looks up.

Usage:
    uv run python pipeline/fetch_drafting_context.py
"""

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


# ─── Section-brief schema ─────────────────────────────────────────────────────────────

class Requirement(BaseModel):
    """A single obligation the section imposes, classified by the kind of change it makes."""

    change_type: fc.ChangeType = Field(
        description=(
            'How this requirement changes existing law or policy: "addition" creates something '
            'that did not exist before (a new requirement, authority, program, definition, or '
            'report); "modification" alters something that already exists (amended text, an '
            'adjusted threshold, a revised definition, narrowed or broadened scope); "deletion" '
            "repeals or removes an existing one."
        )
    )
    description: str = Field(
        description=(
            "The obligation in plain English — the concrete requirement with its triggers, "
            "thresholds, and exceptions folded in, separated from the statutory prose."
        )
    )


class SectionContext(BaseModel):
    ndaa_id: str
    fiscal_year: int
    section_number: str
    section_heading: str
    overview: str = Field(
        description=(
            "Plain-English essence of the section: what it requires and the logic behind it, "
            "with the statutory language stripped away. Orient the reader; don't restate the "
            "statute line by line. Fold in how the cited authorities fit together where that is "
            "needed to make sense of the section."
        )
    )
    requirements: list[Requirement] = Field(
        default_factory=list,
        description=(
            "The discrete obligations the section imposes, one per entry, each classified as an "
            "addition, modification, or deletion."
        ),
    )


# ─── Tools (external fetchers only) ───────────────────────────────────────────────────

# Reuse the manifest stage's research tools, minus the DFARS semantic search: this stage only
# resolves *external* sources and distills the section's logic — deciding where/how the change
# lands in DFARS is the downstream drafting agent's job.
TOOLS = [t for t in fc.TOOLS if t["name"] != "get_dfars_context"]
DISPATCH = {k: v for k, v in fc.DISPATCH.items() if k != "get_dfars_context"}


# ─── Prompt + driver ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal research analyst for U.S. Defense Acquisition Regulation.

You are given the text of one section of a National Defense Authorization Act (NDAA). Downstream,
a drafting agent will implement this section inside the Defense Federal Acquisition Regulation
Supplement (DFARS). The NDAA text often contains references to external documents like U.S. Code,
Public Law, Statutes at Large, other NDAAs, or even other sections of the same NDAA. This makes it
difficult for that drafter to understand the full requirement without resolving these indirect
references and seeing past the statutory language.

Your job is to do exactly that — resolve the external context and extract the logic — so the
downstream agent receives the section's essence along with the raw NDAA section. You do NOT decide where
the change belongs in DFARS or how to draft it; that is the drafting agent's job.

Work in two phases:

1. RESEARCH. If the section cannot be understood on its own, call the
   provided tools to resolve any external reference you need:
   - When the section amends or references a U.S. Code section, fetch it (get_usc_section(title, section, year)) so you
     understand what the text being amended actually says. Always prefer calling the tool with
     the section number without any suffixes  (e.g., ``2403`` and not ``2403-1``). For the year, use the NDAA year.
   - When it references another NDAA (commonly "section NNN of the National Defense Authorization
     Act for Fiscal Year YYYY"), fetch that section (get_ndaa_section, year=YYYY, section=NNN).
   - When it references a public law or a Statutes at Large citation, fetch it.
   Fan out to as many tool calls as you need, but only if its needed to understand the NDAA section completely.
   If a tool call fails or comes back empty, retry it once with corrected arguments when the
   problem looks like a bad argument. If it still fails — or every tool call fails — do not abandon
   the brief: fall back to what the NDAA text itself states and flag the gap (see below).

2. SYNTHESIZE. Turn what you gathered into the section brief:
   - overview: brief the drafter in plain English on what the section requires and the logic
     behind it, with statutory language stripped away. Orient them; don't restate the statute
     line by line. Fold what you learned from the resolved references into this overview — where
     it helps, explain how the cited authorities fit together — rather than listing them out.
   - requirements: the discrete obligations the section imposes — one per entry — pulling the
     concrete requirements, triggers, thresholds, and exceptions out of the prose. Classify each
     one's change_type as exactly one of:
       • "addition": creates something that did not exist before — a new requirement, prohibition,
         authority, program, pilot, definition, or report obligation.
       • "modification": alters something that already exists — amended text, an adjusted threshold
         or dollar figure, a revised definition, or narrowed/broadened scope.
       • "deletion": removes or repeals an existing requirement, authority, or provision.

When you could NOT resolve a reference a requirement depends on (a tool failed, or every tool call
failed), still produce that requirement from the NDAA text alone:
   - change_type: infer it from the NDAA's own amendatory language — "amended", "striking",
     "inserting" → modification; "repealed", "struck out" → deletion; "established", "shall
     submit", a new prohibition, authority, or report → addition.
   - description: state the obligation as far as the NDAA's own words support it — capture the
     amendatory instruction or directive itself in plain English (e.g. "Amends <cited section> by
     striking X and inserting Y"). Do NOT invent the contents of the unresolved reference, and do
     NOT drop the requirement.

After researching, output the section brief in the required structured format."""


def process_section(llm, doc: dict) -> SectionContext:
    """Run the research-then-synthesize loop for one NDAA section document."""
    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fc.build_user_prompt(doc)},
    ]
    conversation = run_tool_loop(llm, input_messages, TOOLS, DISPATCH)
    conversation.append(
        {
            "role": "user",
            "content": (
                "Now output the section brief in the required structured format, using the "
                "context you gathered: the plain-English overview and the discrete requirements "
                "(each classified as an addition, modification, or deletion). If any reference "
                "could not be resolved, still include the requirement from the NDAA text."
            ),
        }
    )
    result = get_structured_response_from_input(llm, conversation, SectionContext)

    # Pin the identity fields from the source doc rather than trusting the model.
    section = doc["section"]
    result.ndaa_id = doc["_id"]
    result.fiscal_year = doc["fiscal_year"]
    result.section_number = str(section["number"])
    result.section_heading = section["heading"]
    return result


def run_one(year: str, section: str) -> None:
    """Process a single section and write its own context JSON file."""
    doc = get_doc_by_year_section(fc._client(), DB, NDAAS, int(year), section)
    if not doc:
        print(f"No NDAA section found for {year}_{section}")
        sys.exit(1)

    print(f"Processing NDAA {year}_{section}: {doc['section'].get('heading', '')}")
    result = process_section(connect_to_openai(), doc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"context_{year}_{section}.json"
    out_path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


def diff_sections(diff_path: Path) -> list[tuple[str, str]]:
    """Read the DFARS diff; return de-duplicated (ndaa_year, ndaa_section) pairs in file order.

    These are exactly the NDAA sections framework2 looks a brief up for (it keys
    dfars_diff_all.json entries by "<year>_<section>"), so sourcing pairs here keeps
    brief coverage aligned with what the drafting pipeline consumes.
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
    run_pairs(diff_sections(DIFF_FILE), "drafting_context_fr_cases.json")
    # run_one("2024", "865")
