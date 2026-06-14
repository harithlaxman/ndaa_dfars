"""
Baseline NDAA -> DFARS Drafting Framework
=========================================
A deliberately dumb baseline for the 1:N framework2 pipeline. For each NDAA it
makes ONE LLM call: hand the model the NDAA text plus every affected DFARS node
(rolled up to section units), and ask it to return the full revised text for each
node. No change manifest, no delegation, no per-section fan-out, no reconciliation.

Its only purpose is to measure how much framework2's machinery actually buys us:
the output is written in the same shape framework2 produces, so framework2's
eval.py (BLEU + section/group judge) scores it unchanged for a head-to-head
comparison.

Inputs:
  - data/dfars_diff_all.json: per NDAA (year, section), the implementing DFARS
    case(s) and the before/after text of every changed DFARS node. Changed nodes
    are rolled up to their enclosing SECTION via framework2's _group_sections.
  - pipeline/out/manifests_fr_cases.json: used ONLY to gate which NDAAs run, so the
    baseline covers the same population framework2 evaluates. Its content is never
    read.
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

_BASELINE_DIR = Path(__file__).resolve().parent
_FRAMEWORK2_DIR = _BASELINE_DIR.parent / "framework2"
_PIPELINE_DIR = _BASELINE_DIR.parents[1]   # pipeline/ -- for `agents.*` imports
_REPO_ROOT = _BASELINE_DIR.parents[2]      # ndaa_dfars/ -- for `utils.*` and data/
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_REPO_ROOT))

# Reuse framework2's node grouping + single-NDAA filter so the baseline draws the
# exact same drafting units and population.
from agents.framework2.agent import _group_sections  # noqa: E402
from agents.framework2.utils import get_single_ndaa_allowed  # noqa: E402
from agents.baseline.schemas import BaselineDraft  # noqa: E402
from utils.mongo_utils import getMongoClient, get_doc_by_year_section  # noqa: E402

_DATA_DIR = _REPO_ROOT / "data"
_DIFF_FILE = _DATA_DIR / "dfars_diff_all.json"
_MANIFEST_FILE = _PIPELINE_DIR / "out" / "manifests_fr_cases.json"
_DRAFTING_GUIDE = (_FRAMEWORK2_DIR / "far_drafting_guide.md").read_text(encoding="utf-8")

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


def _manifest_ndaa_ids() -> set[str]:
    """The set of '<year>_<section>' ids that have a pre-computed manifest.

    Used only to gate which NDAAs the baseline runs on, matching framework2's
    population. The manifest content itself is never read.
    """
    with open(_MANIFEST_FILE) as f:
        return {s["ndaa_id"] for s in json.load(f).get("sections", [])}


def load_baseline_groups(single_ndaa: bool = False) -> list[dict]:
    """Build one group per NDAA from data/dfars_diff_all.json.

    Unlike framework2, the baseline does NOT batch sections into fives -- every
    DFARS section for an NDAA stays in a single group so the one LLM call sees the
    NDAA and all its nodes together.

    Returns a list of dicts, each:
        {
          "ndaa": {"year": str, "section": str, "header": str, "text": str},
          "dfars_sections": [{"section", "part", "subpart", "before", "after"}, ...]
        }
    """
    allowed = get_single_ndaa_allowed() if single_ndaa else None
    manifest_ids = _manifest_ndaa_ids()

    with open(_DIFF_FILE) as f:
        diff: dict = json.load(f)

    client = getMongoClient()
    groups: list[dict] = []
    try:
        for entry in diff.get("sections", []):
            year = str(entry["ndaa_year"])
            section = str(entry["ndaa_section"])
            ndaa_id = f"{year}_{section}"

            # Match framework2's population: only NDAAs with a manifest.
            if ndaa_id not in manifest_ids:
                continue

            changes: list[dict] = []
            for case in entry.get("cases", []):
                changes.extend(case.get("changes", []))
            if not changes:
                continue

            dfars_secs = _group_sections(changes)

            if allowed is not None:
                allowed_secs = allowed.get((year, section))
                if not allowed_secs:
                    continue
                dfars_secs = [
                    s for s in dfars_secs
                    if any(a in s["section"] for a in allowed_secs)
                ]

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
            })
    finally:
        client.close()

    return groups


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert DFARS rulemaking drafter. You are given an NDAA provision and "
    "the current text of the DFARS sections it affects. Revise each DFARS section to "
    "implement what the NDAA mandates.\n\n"
    "Rules:\n"
    "- Make the MINIMUM changes necessary. Preserve every word of existing text that "
    "the NDAA does not require changing.\n"
    "- Return the COMPLETE revised text for each section (not a diff, not a summary).\n"
    "- Keep the existing regulatory structure: subsection numbering, paragraph "
    "hierarchy, and definition placement.\n"
    "- Output regulatory text only -- no markdown, no commentary.\n\n"
    "Follow the FAR/DFARS Drafting Guide conventions below:\n\n" + _DRAFTING_GUIDE
)


def run_baseline(group: dict) -> list[dict]:
    """One LLM call: NDAA + all affected DFARS nodes -> revised node text.

    Returns framework2-compatible section drafts:
        [{"section", "draft_clean", "before", "after"}, ...]
    """
    ndaa = group["ndaa"]
    sections = group["dfars_sections"]

    nodes_block = "\n\n".join(
        f"[{i}] SECTION {s['section']}\n"
        f'"""\n{s["before"]}\n"""'
        for i, s in enumerate(sections)
    )

    prompt = f"""NDAA PROVISION (FY{ndaa['year']}, Section {ndaa['section']} -- {ndaa['header']}):
\"\"\"
{ndaa['text']}
\"\"\"

DFARS SECTIONS TO REVISE (current text):
{nodes_block}

For EACH DFARS section above, return its full revised text implementing the NDAA
provision. Use the exact section number shown (e.g. "{sections[0]['section']}").
Preserve all existing text the NDAA does not require changing.
"""

    llm = _get_llm().with_structured_output(BaselineDraft)
    result: BaselineDraft = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])

    revised_by_section = {d.section: d.revised_text for d in result.sections}

    drafts: list[dict] = []
    for s in sections:
        drafts.append({
            "section": s["section"],
            # Fall back to the original text if the model omitted a section.
            "draft_clean": revised_by_section.get(s["section"], s["before"]),
            "before": s["before"],
            "after": s["after"],
        })
    return drafts


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

    parser = argparse.ArgumentParser(description="Baseline NDAA -> DFARS drafter")
    parser.add_argument("--limit", type=int, default=None, help="Max NDAAs to process")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--single-ndaa", action="store_true",
        help="Only process DFARS sections from documents with exactly one NDAA citation",
    )
    args = parser.parse_args()

    print("Loading NDAA groups (baseline) ...")
    groups = load_baseline_groups(single_ndaa=args.single_ndaa)
    print(f"Found {len(groups)} NDAAs")

    if args.limit:
        groups = groups[: args.limit]

    results: list[dict] = []
    for idx, group in enumerate(groups, 1):
        ndaa = group["ndaa"]
        n = len(group["dfars_sections"])
        print(f"\n[{idx}/{len(groups)}] NDAA {ndaa['year']} s{ndaa['section']} -> {n} DFARS section(s)")

        try:
            drafts = run_baseline(group)
            results.append({
                "ndaa_year": ndaa["year"],
                "ndaa_section": ndaa["section"],
                "ndaa_header": ndaa.get("header", ""),
                "n_dfars_sections": n,
                "section_drafts": drafts,
            })
            print(f"  done -- {len(drafts)} sections drafted")
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({
                "ndaa_year": ndaa["year"],
                "ndaa_section": ndaa["section"],
                "error": str(exc),
            })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(_DATA_DIR / "results" / f"pipeline_baseline_results_{ts}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
