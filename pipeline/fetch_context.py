#!/usr/bin/env python3
"""Turn a single NDAA section into a list of plain-English change manifests.

An NDAA section is dense statutory text that usually can't be understood on its own: it
amends a U.S. Code section, points back to an earlier NDAA, or leans on a public law or
Statutes-at-Large citation. So rather than a one-shot prompt, this stage lets an LLM fan
out to tool calls (GovInfo text fetchers, an "other NDAA section" lookup, and a DFARS
context search) to pull in exactly the context it needs, then draft a list of *atomic*
change manifests: jargon-stripped descriptions of each discrete change the section
mandates, anchored toward later implementing that change inside DFARS.

Reads the section from Mongo (``ndaa_dfars.ndaas``); writes manifests as JSON to
``pipeline/out/``. Never writes to Mongo.

Usage:
    uv run python pipeline/fetch_context.py
"""

import bisect
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from utils.api_utils import (
    get_law_by_bill,
    get_law_by_statute_citation,
    get_public_law,
    get_statute_by_citation,
    get_statute_by_law,
    get_usc_section,
)
from utils.mongo_utils import (
    get_doc_by_year_section,
    getMongoClient,
    vector_search_dfars,
)
from utils.openai import (
    connect_to_openai,
    get_structured_response_from_input,
    run_tool_loop,
)
from pymongo import DESCENDING

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "pipeline" / "out"
DATA_DIR = _ROOT / "data"
DB = "ndaa_dfars"
NDAAS = "ndaas"
DFARS = "dfars"


# ─── Change manifest schema ────────────────────────────────────────────────────────

ChangeType = Literal[
    "addition",
    "modification",
    "deletion",
]


class ChangeManifest(BaseModel):
    change_id: int = Field(description="1-based ordinal of this change within the section")
    change_type: ChangeType = Field(
        description=(
            'The kind of change this manifest captures: "addition" creates a new '
            'requirement/authority/program, "modification" alters an existing one '
            '(including threshold or definition changes), "deletion" repeals or removes one'
        )
    )
    description: str = Field(
        description=(
            "Plain-English description of what this change is trying to do — its purpose and "
            "intended effect, with statutory jargon stripped. Explain the goal the change is "
            "reaching for, not just a condensed restatement of the statutory text."
        )
    )
    subject: str = Field(description="Short topic label for the change")
    applies_to: Optional[str] = Field(
        default=None, description="Who or what the change binds, in plain English"
    )
    conditions: Optional[str] = Field(
        default=None, description="Triggers, thresholds, or exceptions that gate the change"
    )
    effective_date: Optional[str] = Field(
        default=None, description="Effective date or deadline as stated, else null"
    )
    amends: list[str] = Field(
        default_factory=list,
        description='Normalized authorities this change amends, e.g. "10 U.S.C. 2304"',
    )
    source_authorities: list[str] = Field(
        default_factory=list,
        description="Other citations relied on to understand the change",
    )
    dfars_implementation_hint: Optional[str] = Field(
        default=None, description="Candidate DFARS/FAR location(s) to implement the change, else null"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Ambiguities a downstream implementer would need to resolve",
    )


class SectionManifests(BaseModel):
    ndaa_id: str
    fiscal_year: int
    section_number: str
    section_heading: str
    manifests: list[ChangeManifest] = Field(default_factory=list)


# ─── Mongo-backed tools ─────────────────────────────────────────────────────────────

# A single client shared by the Mongo-backed tools, opened lazily on first use.
_mongo_client = None
_latest_dfars_version = None
_snapshot_dates = None
# DFARS version that get_dfars_context should search for the section currently being
# processed. Set per section (to the snapshot in effect just before the rule's effective
# date, effective_on) when that date is known; None falls back to the latest snapshot.
_dfars_version = None


def _client():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = getMongoClient()
    return _mongo_client


def get_ndaa_section(year: int, section: str) -> Optional[str]:
    """Return the heading + text of another NDAA section, or None if not found."""
    doc = get_doc_by_year_section(_client(), DB, NDAAS, int(year), str(section))
    if not doc:
        return None
    sec = doc.get("section", {})
    return f"{sec.get('number', '')} {sec.get('heading', '')}\n\n{sec.get('text', '')}".strip()


def _latest_dfars_date():
    """Resolve (and cache) the most recent DFARS snapshot's version_date."""
    global _latest_dfars_version
    if _latest_dfars_version is None:
        coll = _client()[DB][DFARS]
        doc = coll.find_one(sort=[("version_date", DESCENDING)])
        _latest_dfars_version = doc["version_date"] if doc else None
    return _latest_dfars_version


def _snapshot_versions():
    """Resolve (and cache) every DFARS snapshot version_date, sorted ascending."""
    global _snapshot_dates
    if _snapshot_dates is None:
        _snapshot_dates = sorted(_client()[DB][DFARS].distinct("version_date"))
    return _snapshot_dates


def _dfars_version_before(effective: datetime):
    """The DFARS snapshot in effect just before a rule's ``effective`` date.

    A rule taking effect on ``effective`` first appears in the earliest snapshot dated
    on or after it, so the version that does *not* yet contain the change is the snapshot
    immediately before that one — equivalently, the latest snapshot dated before
    ``effective``. Returns None when no earlier snapshot exists.
    """
    dates = _snapshot_versions()
    i = bisect.bisect_left(dates, effective)
    return dates[i - 1] if i > 0 else None


def get_dfars_context(query: str, limit: int = 5) -> Optional[list]:
    """Semantically search the latest DFARS snapshot for nodes relevant to a topic.

    Uses Atlas Vector Search (auto-embedding) over the DFARS snapshot in effect just
    before the rule's effective date, falling back to the latest snapshot when that
    date is unknown. Returns up to ``limit``
    ``{section_number, heading, excerpt}`` matches so the model can see where and how DFARS
    speaks to the change, or None if nothing matches.
    """
    version = _dfars_version if _dfars_version is not None else _latest_dfars_date()
    if version is None:
        return None
    docs = vector_search_dfars(
        _client(), DB, DFARS, str(query), version, limit=int(limit),
        index_name="autoembed_index",
    )
    results = []
    for doc in docs:
        sec = doc.get("section", {})
        text = sec.get("text", "") or ""
        results.append(
            {
                "section_number": doc.get("section_number"),
                "heading": sec.get("heading", ""),
                "excerpt": text[:1500],
            }
        )
    return results or None


# ─── Tool schemas + dispatch ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "name": "get_usc_section",
        "description": "Fetch the current text of a United States Code section (e.g. title 10, section 2304).",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "integer", "description": "U.S. Code title number, e.g. 10"},
                "section": {"type": "string", "description": 'Section number, e.g. "2304" (may include a hyphen)'},
                "year": {"type": "integer", "description": "Optional edition year; omit for most recent"},
            },
            "required": ["title", "section"],
        },
    },
    {
        "type": "function",
        "name": "get_public_law",
        "description": "Fetch the text of a public or private law by Congress, type, and number, e.g. Public Law 111-78.",
        "parameters": {
            "type": "object",
            "properties": {
                "congress": {"type": "integer", "description": "Congress number, e.g. 111"},
                "lawnum": {"type": "integer", "description": "Law number within that Congress, e.g. 78"},
                "lawtype": {"type": "string", "enum": ["public", "private"], "description": 'Defaults to "public"'},
            },
            "required": ["congress", "lawnum"],
        },
    },
    {
        "type": "function",
        "name": "get_law_by_bill",
        "description": "Fetch the text of the law enacted from a given bill, e.g. congress 111, bill 'S. 3397'.",
        "parameters": {
            "type": "object",
            "properties": {
                "congress": {"type": "integer", "description": "Congress number, e.g. 111"},
                "bill_number": {"type": "string", "description": 'Bill number, e.g. "S. 3397" or "H.R. 2544"'},
            },
            "required": ["congress", "bill_number"],
        },
    },
    {
        "type": "function",
        "name": "get_law_by_statute_citation",
        "description": "Fetch the text of a law by its Statutes at Large citation, e.g. '124 Stat 2859'.",
        "parameters": {
            "type": "object",
            "properties": {
                "citation": {"type": "string", "description": 'Statutes at Large citation, e.g. "124 Stat 2859"'},
            },
            "required": ["citation"],
        },
    },
    {
        "type": "function",
        "name": "get_statute_by_law",
        "description": "Fetch a Statutes at Large document by Congress, law type, and number.",
        "parameters": {
            "type": "object",
            "properties": {
                "congress": {"type": "integer", "description": "Congress number, e.g. 108"},
                "lawnum": {"type": "integer", "description": "Law number within that Congress, e.g. 481"},
                "lawtype": {"type": "string", "enum": ["public", "private"], "description": 'Defaults to "public"'},
            },
            "required": ["congress", "lawnum"],
        },
    },
    {
        "type": "function",
        "name": "get_statute_by_citation",
        "description": "Fetch a Statutes at Large document by volume and page, e.g. volume 118, page 3910.",
        "parameters": {
            "type": "object",
            "properties": {
                "volume": {"type": "integer", "description": "Volume number, e.g. 118"},
                "page": {"type": "integer", "description": "Page number within the volume, e.g. 3910"},
            },
            "required": ["volume", "page"],
        },
    },
    {
        "type": "function",
        "name": "get_ndaa_section",
        "description": "Fetch the heading and text of another NDAA section, by fiscal year and section number.",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Fiscal year of the NDAA, e.g. 2017"},
                "section": {"type": "string", "description": 'Section number, e.g. "847"'},
            },
            "required": ["year", "section"],
        },
    },
    {
        "type": "function",
        "name": "get_dfars_context",
        "description": (
            "Semantically search the current DFARS for sections relevant to a topic, to see where and how "
            "DFARS already addresses it. Returns the top matching section numbers, headings, and excerpts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A natural-language description of the topic to look up in DFARS"},
            },
            "required": ["query"],
        },
    },
]

DISPATCH = {
    "get_usc_section": get_usc_section,
    "get_public_law": get_public_law,
    "get_law_by_bill": get_law_by_bill,
    "get_law_by_statute_citation": get_law_by_statute_citation,
    "get_statute_by_law": get_statute_by_law,
    "get_statute_by_citation": get_statute_by_citation,
    "get_ndaa_section": get_ndaa_section,
    "get_dfars_context": get_dfars_context,
}


# ─── Prompt + driver ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal-change analyst for U.S. defense acquisition regulation.

You are given the text of one section of a National Defense Authorization Act (NDAA). Your job
is to break it down into a list of atomic CHANGE MANIFESTS: each manifest captures a single,
discrete change the section mandates, described in plain English with the statutory jargon
stripped away. These manifests will later be used to implement the section inside the Defense
Federal Acquisition Regulation Supplement (DFARS).

Work in two phases:

1. RESEARCH. The section often cannot be fully understood on its own. Before drafting, call the
   provided tools to resolve any context you need:
   - When the section amends or references a U.S. Code section, fetch it (get_usc_section) so you
     understand what the text being amended actually says.
   - When it references another NDAA (commonly "section NNN of the National Defense Authorization
     Act for Fiscal Year YYYY"), fetch that section (get_ndaa_section, year=YYYY, section=NNN).
   - When it references a public law or a Statutes at Large citation, fetch it.
   - Use get_dfars_context to see whether and where DFARS already addresses the topic; this helps
     you fill dfars_implementation_hint.
   Fan out to as many tool calls as you need, but only those you actually need.

2. DRAFT. Produce one manifest per discrete change. Splitting rules:
   - A section that creates a requirement, sets a threshold, AND mandates a report is THREE
     manifests, not one.
   - A pure conforming/technical amendment (e.g. striking and re-inserting a cross-reference with
     no substantive effect) does not need its own manifest unless it changes meaning.
   Classify each manifest's change_type as exactly one of:
   - "addition": the change creates something that did not exist before — a new requirement,
     prohibition, authority, program, pilot, definition, or report obligation.
   - "modification": the change alters something that already exists — amending statutory text,
     adjusting a threshold or dollar figure, revising a definition, or extending/narrowing scope.
   - "deletion": the change removes or repeals an existing requirement, authority, or provision.
   For each manifest: write description in clear plain English, explaining what the change is
   trying to do — its purpose and intended effect, not just a condensed restatement of the text;
   set amends to the normalized
   authorities the change modifies (e.g. "10 U.S.C. 2304"); set source_authorities to other cites
   you relied on; suggest a dfars_implementation_hint when you can; and record genuine ambiguities
   in open_questions.

After researching, output the manifests in the required structured format."""


def build_user_prompt(doc: dict) -> str:
    section = doc["section"]
    return (
        f"NDAA Fiscal Year: {doc['fiscal_year']}\n"
        f"Section Number: {section['number']}\n"
        f"Section Heading: {section['heading']}\n\n"
        f"Section Text:\n{section['text']}"
    )


def process_section(llm, doc: dict) -> SectionManifests:
    """Run the research-then-draft loop for one NDAA section document."""
    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(doc)},
    ]
    conversation = run_tool_loop(llm, input_messages, TOOLS, DISPATCH)
    conversation.append(
        {
            "role": "user",
            "content": (
                "Now output the final list of change manifests for this section in the required "
                "structured format, using the context you gathered."
            ),
        }
    )
    result = get_structured_response_from_input(llm, conversation, SectionManifests)

    # Pin the identity fields from the source doc rather than trusting the model.
    section = doc["section"]
    result.ndaa_id = doc["_id"]
    result.fiscal_year = doc["fiscal_year"]
    result.section_number = str(section["number"])
    result.section_heading = section["heading"]
    return result


def csv_sections(csv_path: Path) -> list[tuple[str, str]]:
    """Read a CSV with ndaa_year/ndaa_section columns; return de-duplicated (year, section)
    pairs in file order."""
    pairs: list[tuple[str, str]] = []
    seen = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            year = (row.get("ndaa_year") or "").strip()
            section = (row.get("ndaa_section") or "").strip()
            if not year or not section:
                continue
            key = (year, section)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def run_one(year: str, section: str) -> None:
    """Process a single section and write its own manifests JSON file."""
    doc = get_doc_by_year_section(_client(), DB, NDAAS, int(year), section)
    if not doc:
        print(f"No NDAA section found for {year}_{section}")
        sys.exit(1)

    print(f"Processing NDAA {year}_{section}: {doc['section'].get('heading', '')}")
    result = process_section(connect_to_openai(), doc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"manifests_{year}_{section}.json"
    out_path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"Wrote {out_path} — {len(result.manifests)} manifests")


def csv_effective_dates(csv_path: Path) -> dict[tuple[str, str], datetime]:
    """Map (year, section) -> rule effective date for rows that carry one.

    effective_on is stored as ``YYYY-MM-DD``; rows without a parseable date are skipped.
    """
    out: dict[tuple[str, str], datetime] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            year = (row.get("ndaa_year") or "").strip()
            section = (row.get("ndaa_section") or "").strip()
            raw = (row.get("effective_on") or "").strip()
            if not (year and section and raw):
                continue
            try:
                out[(year, section)] = datetime.strptime(raw, "%Y-%m-%d")
            except ValueError:
                continue
    return out


def run_csv(csv_path: Path, out_name: str) -> None:
    """Process every unique NDAA section in a CSV into one combined JSON file."""
    global _dfars_version
    pairs = csv_sections(csv_path)
    effective_dates = csv_effective_dates(csv_path)
    print(f"{csv_path.name} has {len(pairs)} unique NDAA sections")
    llm = connect_to_openai()
    client = _client()

    sections, not_found, failed = [], [], []
    for i, (year, section) in enumerate(pairs, 1):
        tag = f"{year}_{section}"
        doc = get_doc_by_year_section(client, DB, NDAAS, int(year), section)
        if not doc:
            print(f"[{i}/{len(pairs)}] {tag}: not found in Mongo, skipping")
            not_found.append(tag)
            continue
        # Search the DFARS snapshot in effect just before this section's rule took
        # effect; None falls back to the latest snapshot.
        eff = effective_dates.get((year, section))
        _dfars_version = _dfars_version_before(eff) if eff else None
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
    run_csv(DATA_DIR / "fr_cases.csv", "manifests_fr_cases.json")


if __name__ == "__main__":
    main()
