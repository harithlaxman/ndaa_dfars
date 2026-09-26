"""
Framework1 NDAA -> DFARS Drafting Framework
===========================================
A single-call NDAA -> DFARS drafter. For each NDAA it makes ONE LLM call: hand the
model the NDAA text, the section's resolved external citations/references, and every
affected DFARS node (rolled up to section units), and ask it, for each node, how it
changes (added/modified/deleted) and its full revised text. No change manifest, no
delegation, no per-section fan-out, no reconciliation.

It is the baseline plus one thing: the external citation context produced by
pipeline/fetch_reference_context.py is folded into the drafting prompt, so we can
measure what resolving those cross-references alone buys.

Its only purpose is to measure how much framework2's machinery actually buys us.
Unlike framework2's per-node output, the framework1 result is saved per NDAA as
three concatenated, section-labeled blobs -- `before` (original text), `draft`
(the model's output), and `after` (ground truth) -- so a whole-draft eval can
compare `draft` against `after`. Deleted sections render as "[REMOVED]".

Inputs:
  - data/dfars_diff_all.json: per NDAA (year, section), the implementing DFARS
    case(s) and the before/after text of every changed DFARS node. Changed nodes
    are rolled up to their enclosing SECTION via framework2's _group_sections.
  - pipeline/out/reference_context_fr_cases.json: the resolved external citations
    per NDAA section (from pipeline/fetch_reference_context.py), keyed by
    "<year>_<section>". Optional -- if absent, framework1 degrades to the baseline.
  - Mongo (db "ndaa_dfars", collection "ndaas"): the NDAA section's full statutory
    text, fetched per NDAA. Read-only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

_FRAMEWORK1_DIR = Path(__file__).resolve().parent
_FRAMEWORK2_DIR = _FRAMEWORK1_DIR.parent / "framework2"
_PIPELINE_DIR = _FRAMEWORK1_DIR.parents[1]   # pipeline/ -- for `agents.*` imports
_REPO_ROOT = _FRAMEWORK1_DIR.parents[2]      # ndaa_dfars/ -- for `utils.*` and data/
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_REPO_ROOT))

# Reuse framework2's node grouping so framework1 draws the exact same drafting units.
from agents.framework2.agent import _group_sections  # noqa: E402
from agents.framework1.schemas import Framework1Draft  # noqa: E402
from utils.mongo_utils import getMongoClient, get_doc_by_year_section  # noqa: E402

_DATA_DIR = _REPO_ROOT / "data"
_DIFF_FILE = _DATA_DIR / "dfars_diff_all.json"
_DRAFTING_GUIDE = (_PIPELINE_DIR / "far_drafting_guide.md").read_text(encoding="utf-8")
# Resolved external citations/references per NDAA section (from
# pipeline/fetch_reference_context.py), keyed by "<year>_<section>". This is what
# distinguishes framework1 from the baseline: the same single drafting call, but
# with each section's external dependencies resolved and folded into the prompt.
_REFERENCE_CONTEXT_FILE = _PIPELINE_DIR / "out" / "reference_context_fr_cases.json"

DB = "ndaa_dfars"
NDAAS = "ndaas"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    # Larger max_tokens than framework2 (4096): one call returns every revised node.
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        azure_endpoint=os.environ.get(
            "OPENAI_ENDPOINT", os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        ),
        api_key=os.environ.get(
            "OPENAI_API_KEY", os.environ.get("AZURE_OPENAI_API_KEY", "")
        ),
        api_version="2025-03-01-preview",
        temperature=temperature,
        max_tokens=16000,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_framework1_groups() -> list[dict]:
    """Build one group per NDAA from data/dfars_diff_all.json.

    Unlike framework2, framework1 does NOT batch sections into fives -- every
    DFARS section for an NDAA stays in a single group so the one LLM call sees the
    NDAA and all its nodes together.

    Returns a list of dicts, each:
        {
          "ndaa": {"year": str, "section": str, "header": str, "text": str},
          "dfars_sections": [{"section", "part", "subpart", "before", "after"}, ...],
          "dependencies": [{"reference": str, "explanation": str}, ...]
        }
    """

    with open(_DIFF_FILE) as f:
        diff: dict = json.load(f)

    # Resolved external references per NDAA section, keyed by "<year>_<section>".
    # Optional: if the context file isn't there yet, framework1 still runs -- it
    # just degrades to the baseline (no external citation context in the prompt).
    refs_by_id: dict[str, list[dict]] = {}
    if _REFERENCE_CONTEXT_FILE.exists():
        with open(_REFERENCE_CONTEXT_FILE) as f:
            refs_by_id = {
                s["ndaa_id"]: s.get("dependencies", [])
                for s in json.load(f).get("sections", [])
            }
    else:
        print(
            f"  warning: no reference context at {_REFERENCE_CONTEXT_FILE}; "
            "drafting without external citation context"
        )

    client = getMongoClient()
    groups: list[dict] = []
    try:
        for entry in diff.get("sections", []):
            year = str(entry["ndaa_year"])
            section = str(entry["ndaa_section"])

            changes: list[dict] = []
            for case in entry.get("cases", []):
                changes.extend(case.get("changes", []))
            if not changes:
                continue

            dfars_secs = _group_sections(changes)

            if not dfars_secs:
                continue
            if len(dfars_secs) > 25:
                print(f"  skip NDAA {year} s{section}: {len(dfars_secs)} DFARS sections (>25)")
                continue

            ndaa_doc = get_doc_by_year_section(client, DB, NDAAS, year, section)
            if ndaa_doc is None:
                print(f"  skip NDAA {year} s{section}: not found in Mongo '{NDAAS}'")
                continue
            ndaa_section = ndaa_doc.get("section", {})

            groups.append({
                "ndaa": {
                    "year": year,
                    "section": section,
                    "header": ndaa_section.get("heading", ""),
                    "text": ndaa_section.get("text", ""),
                },
                "dfars_sections": dfars_secs,
                "dependencies": refs_by_id.get(f"{year}_{section}", []),
            })
    finally:
        client.close()

    return groups


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert DFARS rulemaking drafter. You are given an NDAA provision and "
    "the current text of the DFARS sections it affects. Implement what the NDAA "
    "mandates in DFARS.\n\n"
    "For each affected DFARS section, decide how it must change and label it with a "
    "change_type:\n"
    "- 'modified' -- the section's existing text must change. Make the MINIMUM edits "
    "necessary, preserve every word the NDAA does not require changing, and return "
    "the COMPLETE revised text. Adding one or more new paragraphs to an existing "
    "section counts as modifying that section.\n"
    "- 'deleted' -- the NDAA removes the section (or its sole requirement) in its "
    "entirety. Leave the revised text empty.\n"
    "- 'added' -- a brand-new section, subsection, or clause is required. Create a "
    "new node ONLY if the mandate genuinely cannot be implemented by modifying an "
    "existing section. When you do, assign its number and heading following the "
    "FAR/DFARS Drafting Guide rules for parts, subparts, sections, and subsections, "
    "and return its full text.\n\n"
    "General rules:\n"
    "- Strongly prefer modifying existing sections over creating new ones.\n"
    "- Return the COMPLETE revised text for each section (not a diff, not a summary).\n"
    "- Preserve existing regulatory structure: subsection numbering, paragraph "
    "hierarchy, and definition placement, unless the mandate requires changing it.\n"
    "- Output regulatory text only -- no markdown, no commentary.\n\n"
    "Follow the FAR/DFARS Drafting Guide conventions below:\n\n" + _DRAFTING_GUIDE
)


# Marker for a section removed in its entirety, mirrored from the ground-truth
# "after" text in data/dfars_diff_all.json.
_REMOVED_MARKER = "[REMOVED]"


def _blob(items: list[tuple[str, str]]) -> str:
    """Concatenate (section_number, text) pairs into one section-labeled blob."""
    return "\n\n".join(f"{sec}\n{text.strip()}" for sec, text in items)


def _format_dependencies(deps: list[dict]) -> str:
    """Render resolved external references as a compact reference -> explanation list."""
    lines = [
        f"- {d.get('reference', '').strip()}: {d.get('explanation', '').strip()}"
        for d in deps
        if d.get("reference") or d.get("explanation")
    ]
    return "\n".join(lines) if lines else "(no external references resolved)"


def run_framework1(group: dict) -> dict:
    """One LLM call: NDAA + all affected DFARS nodes -> revised node text.

    Returns the per-NDAA result as concatenated, section-labeled blobs plus the
    per-section change classification:
        {"before": str, "draft": str, "after": str,
         "section_changes": [{"section", "change_type"}, ...]}
    """
    ndaa = group["ndaa"]
    sections = group["dfars_sections"]
    dependencies = group.get("dependencies", [])

    nodes_block = "\n\n".join(
        f"[{i}] SECTION {s['section']}\n"
        f'"""\n{s["before"]}\n"""'
        for i, s in enumerate(sections)
    )

    prompt = f"""NDAA PROVISION (FY{ndaa['year']}, Section {ndaa['section']} -- {ndaa['header']}):
\"\"\"
{ndaa['text']}
\"\"\"

EXTERNAL CONTEXT (the section's citations and references, resolved -- each is what
the cited authority says and what this NDAA section actually changes about it):
{_format_dependencies(dependencies)}

Below are the list of DFARS sections that might be impacted by the NDAA section.

DFARS SECTIONS (current text):
{nodes_block}

For each DFARS section above, implement the NDAA provision: return its change_type
('modified', 'deleted', or 'added') and the full revised text. Use the exact
section number shown (e.g. "{sections[0]['section']}") for sections you modify or
delete. Only if the mandate cannot be implemented in any existing section, add a
new section with a new number per the Drafting Guide. Preserve all existing text
the NDAA does not require changing.
"""

    llm = _get_llm().with_structured_output(Framework1Draft)
    result: Framework1Draft = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])

    # Ground-truth blobs over the input sections (deleted sections already carry
    # the "[REMOVED]" marker in their `after` text).
    in_sorted = sorted(sections, key=lambda s: s["section"])
    before_blob = _blob([(s["section"], s["before"]) for s in in_sorted])
    after_blob = _blob([(s["section"], s["after"]) for s in in_sorted])

    # Draft blob: exactly what the model produced, in section-number order. Deleted
    # sections render as the removal marker; added sections appear under their new
    # number. A section the model drops simply does not appear -- so a whole-draft
    # eval penalizes the omission rather than masking it with the original text.
    # (The model often labels a unit by its actual subsection number, e.g. emits
    # "236.606-70" for the rolled-up "236.606" input unit, so we cannot align the
    # draft to input keys without spuriously duplicating sections.)
    draft_items: list[tuple[str, str]] = []
    section_changes: list[dict] = []
    for d in sorted(result.sections, key=lambda x: x.section):
        text = _REMOVED_MARKER if d.change_type == "deleted" else d.revised_text
        draft_items.append((d.section, text))
        section_changes.append({"section": d.section, "change_type": d.change_type})
    draft_blob = _blob(draft_items)

    return {
        "before": before_blob,
        "draft": draft_blob,
        "after": after_blob,
        "section_changes": section_changes,
    }


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    from dotenv import load_dotenv

    # OPENAI_* keys live in framework2/.env; MONGO_CLIENT_URI comes from the
    # environment (getMongoClient also calls load_dotenv()).
    load_dotenv(_FRAMEWORK2_DIR / ".env")

    parser = argparse.ArgumentParser(description="Framework1 NDAA -> DFARS drafter")
    parser.add_argument("--limit", type=int, default=None, help="Max NDAAs to process")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("Loading NDAA groups (framework1) ...")
    groups = load_framework1_groups()
    print(f"Found {len(groups)} NDAAs")

    if args.limit:
        groups = groups[: args.limit]

    results: list[dict] = []
    for idx, group in enumerate(groups, 1):
        ndaa = group["ndaa"]
        n = len(group["dfars_sections"])
        print(f"\n[{idx}/{len(groups)}] NDAA {ndaa['year']} s{ndaa['section']} -> {n} DFARS section(s)")

        try:
            drafted = run_framework1(group)
            results.append({
                "ndaa_year": ndaa["year"],
                "ndaa_section": ndaa["section"],
                "ndaa_header": ndaa.get("header", ""),
                "n_dfars_sections": n,
                "before": drafted["before"],
                "draft": drafted["draft"],
                "after": drafted["after"],
                "section_changes": drafted["section_changes"],
            })
            print(f"  done -- {len(drafted['section_changes'])} section change(s)")
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({
                "ndaa_year": ndaa["year"],
                "ndaa_section": ndaa["section"],
                "error": str(exc),
            })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(_DATA_DIR / "results" / f"pipeline_framework1_results_{ts}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
