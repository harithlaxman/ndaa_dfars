#!/usr/bin/env python3
"""Brief a DFARS drafter on one NDAA section: gather the *context* it needs, not a change list.

This is a sibling of ``fetch_context.py``. Where that stage decomposes a section into atomic
CHANGE MANIFESTS (the discrete edits to make), this stage answers a different question: given
that some agent now has to draft the DFARS implementation of this NDAA section, what supporting
context does it need that the bare NDAA + DFARS "before" text does not already give it?

So it runs the same research loop (GovInfo text fetchers, an "other NDAA section" lookup, and a
DFARS context search) but synthesizes the findings into a DRAFTING CONTEXT PACK: a plain-English
overview, the resolved statutory backdrop behind the section's cross-references, the existing
DFARS provisions the implementation has to fit against, the defined terms to use consistently,
and concrete drafting considerations. It deliberately does NOT enumerate the changes — that is
``fetch_context.py``'s job, and a drafting agent can consume both side by side.

Reads the section from Mongo (``ndaa_dfars.ndaas``); writes the context pack as JSON to
``pipeline/out/``. Never writes to Mongo.

Usage:
    uv run python pipeline/fetch_drafting_context.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# Reuse the research tooling, DFARS-version pinning, and CSV helpers from the manifest
# stage rather than duplicating ~200 lines of tool schemas. Both files live in pipeline/,
# so the script's own directory is already on sys.path; add it explicitly to be safe.
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
DB = fc.DB
NDAAS = fc.NDAAS


# ─── Drafting-context schema ────────────────────────────────────────────────────────

ReferenceKind = Literal[
    "us_code",
    "public_law",
    "statutes_at_large",
    "ndaa",
    "far",
    "dfars",
    "other",
]


class ResolvedReference(BaseModel):
    """A cross-reference the section leans on, resolved into something the drafter can use."""

    citation: str = Field(
        description='Normalized authority, e.g. "10 U.S.C. 2304" or "Section 847 of the NDAA for FY2017"'
    )
    kind: ReferenceKind = Field(description="What sort of authority this citation points to")
    summary: str = Field(
        description=(
            "Plain-English summary of what the referenced text actually says — enough that the "
            "drafter understands the backdrop without fetching it themselves."
        )
    )
    relevance: str = Field(
        description="Why this matters for drafting the DFARS implementation — how it bears on the change"
    )


class DfarsTouchpoint(BaseModel):
    """An existing DFARS provision the implementation has to fit against."""

    section_number: str = Field(description='DFARS section/clause number, e.g. "252.204-7012"')
    heading: str = Field(default="", description="Section heading, if known")
    relationship: str = Field(
        description=(
            "How this existing provision relates to the work: a candidate place to implement the "
            "change, an adjacent provision the new text must stay consistent with, the host of a "
            "definition to cross-reference, etc."
        )
    )


class KeyTerm(BaseModel):
    """A defined term the drafter must carry through consistently."""

    term: str = Field(description="The defined term")
    definition: str = Field(description="Its operative meaning, in plain English")
    source: Optional[str] = Field(
        default=None, description="Where the term is defined (statute/regulation), if identifiable"
    )


class SectionContext(BaseModel):
    ndaa_id: str
    fiscal_year: int
    section_number: str
    section_heading: str
    overview: str = Field(
        description=(
            "Plain-English briefing: what this NDAA section is doing and what a DFARS "
            "implementation of it has to accomplish. Orient the drafter, don't restate the statute."
        )
    )
    statutory_background: str = Field(
        default="",
        description=(
            "How the cited authorities fit together — the statutory backdrop a drafter needs to "
            "make sense of the amendment (what is being amended, by what, and why it's structured "
            "the way it is). Empty if the section stands on its own."
        ),
    )
    resolved_references: list[ResolvedReference] = Field(
        default_factory=list,
        description="Cross-references resolved into usable summaries (from the research tools)",
    )
    dfars_touchpoints: list[DfarsTouchpoint] = Field(
        default_factory=list,
        description="Existing DFARS provisions the implementation must fit against or build on",
    )
    key_terms: list[KeyTerm] = Field(
        default_factory=list,
        description="Defined terms the drafter must use consistently",
    )
    drafting_considerations: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, actionable notes for the drafter: where the change likely belongs, "
            "conventions to match, pitfalls to avoid, cross-references to wire up."
        ),
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Genuine ambiguities the drafter will have to resolve",
    )


# ─── Prompt + driver ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal research analyst for U.S. defense acquisition regulation.

You are given the text of one section of a National Defense Authorization Act (NDAA). Downstream,
a drafting agent will implement this section inside the Defense Federal Acquisition Regulation
Supplement (DFARS). That agent already has the NDAA text and the current ("before") text of the
DFARS sections it will edit. Your job is to assemble the EXTRA CONTEXT it needs around that — a
DRAFTING CONTEXT PACK — so it can draft accurately without going and resolving every cross-
reference itself.

You are NOT enumerating the changes. Do not produce a change-by-change list of edits; a separate
stage does that. Your deliverable is background and orientation for the drafter.

Work in two phases:

1. RESEARCH. The section usually cannot be understood on its own. Before writing, call the
   provided tools to resolve any context you need:
   - When the section amends or references a U.S. Code section, fetch it (get_usc_section) so you
     understand what the text being amended actually says.
   - When it references another NDAA (commonly "section NNN of the National Defense Authorization
     Act for Fiscal Year YYYY"), fetch that section (get_ndaa_section, year=YYYY, section=NNN).
   - When it references a public law or a Statutes at Large citation, fetch it.
   - Use get_dfars_context to find the existing DFARS provisions this implementation will touch or
     have to stay consistent with — these become your dfars_touchpoints.
   Fan out to as many tool calls as you need, but only those you actually need.

2. SYNTHESIZE. Turn what you gathered into the context pack:
   - overview: brief the drafter in plain English on what the section does and what implementing it
     in DFARS has to accomplish. Orient them; don't restate the statute line by line.
   - statutory_background: explain how the cited authorities fit together — what is being amended,
     by what, and the structure behind it — so the amendment makes sense. Leave empty if the
     section truly stands alone.
   - resolved_references: one entry per cross-reference you resolved, each with a plain-English
     summary of what it says and why it matters here. This is the payoff of the research step —
     the drafter should not need to re-fetch these.
   - dfars_touchpoints: the existing DFARS sections/clauses the implementation must fit against,
     each with how it relates (candidate implementation site, provision to stay consistent with,
     host of a definition to cross-reference, etc.).
   - key_terms: defined terms the drafter must carry through consistently, with their operative
     meaning and source.
   - drafting_considerations: concrete, actionable notes — where the change likely belongs, DFARS
     conventions to match, pitfalls, cross-references to wire up.
   - open_questions: genuine ambiguities the drafter will have to resolve.

After researching, output the context pack in the required structured format."""


def process_section(llm, doc: dict) -> SectionContext:
    """Run the research-then-synthesize loop for one NDAA section document."""
    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fc.build_user_prompt(doc)},
    ]
    conversation = run_tool_loop(llm, input_messages, fc.TOOLS, fc.DISPATCH)
    conversation.append(
        {
            "role": "user",
            "content": (
                "Now output the drafting context pack for this section in the required structured "
                "format, using the context you gathered. Remember: orient and brief the drafter — "
                "do not enumerate the individual changes."
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


def run_csv(csv_path: Path, out_name: str) -> None:
    """Process every unique NDAA section in a CSV into one combined JSON file."""
    pairs = fc.csv_sections(csv_path)
    effective_dates = fc.csv_effective_dates(csv_path)
    print(f"{csv_path.name} has {len(pairs)} unique NDAA sections")
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
        # Search the DFARS snapshot in effect just before this section's rule took
        # effect; None falls back to the latest snapshot. get_dfars_context reads this
        # module-global on fetch_context, so set it there.
        eff = effective_dates.get((year, section))
        fc._dfars_version = fc._dfars_version_before(eff) if eff else None
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


def main() -> None:
    run_csv(DATA_DIR / "fr_cases.csv", "drafting_context_fr_cases.json")


if __name__ == "__main__":
    main()
