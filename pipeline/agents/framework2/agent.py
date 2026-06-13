"""
1:N NDAA -> DFARS Drafting Pipeline
====================================
Three-phase LangGraph pipeline that processes one NDAA section against
N DFARS sections simultaneously:

  Phase 1 -- Delegation (once per NDAA group)
    delegation: assign the pre-computed change manifest to DFARS sections

  Phase 2 -- Per-Section Drafting (PARALLEL per DFARS section via Send())
    Each section gets its own subagent: change_list -> drafting

  Phase 3 -- Reconciliation (once per NDAA group)
    Reviews all N drafts together for consistency

Inputs:
  - data/dfars_diff_all.json: per NDAA (year, section), the implementing DFARS
    case(s) and the before/after text of every changed DFARS node (produced by
    dfars/dfars_diff.py from dated eCFR snapshots in Mongo). Changed nodes are
    rolled up to their enclosing SECTION (part-252 clauses kept whole) so each
    drafting unit carries full-section context.
  - pipeline/out/manifests_fr_cases.json: the pre-computed change manifest per NDAA
    section (from pipeline/fetch_context.py). Sections without one are skipped.
"""

from __future__ import annotations

import json
import operator
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FRAMEWORK_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _FRAMEWORK_DIR.parents[1]   # pipeline/ -- for `agents.*` imports
_REPO_ROOT = _FRAMEWORK_DIR.parents[2]      # ndaa_dfars/ -- for `utils.*` and data/
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from agents.framework2.schemas import DelegationPlan  # noqa: E402
from agents.framework2.utils import get_single_ndaa_allowed  # noqa: E402

_DATA_DIR = _REPO_ROOT / "data"
# Per NDAA section: the implementing DFARS case(s) and the before/after text of
# every changed DFARS node, produced by dfars/dfars_diff.py from dated eCFR
# snapshots in Mongo.
_DIFF_FILE = _DATA_DIR / "dfars_diff_all.json"
_MANIFEST_FILE = _PIPELINE_DIR / "out" / "manifests_fr_cases.json"
_DRAFTING_GUIDE_PATH = _FRAMEWORK_DIR / "far_drafting_guide.md"
_DRAFTING_GUIDE = _DRAFTING_GUIDE_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# LLM & tools
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
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
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Draft post-processing
# ---------------------------------------------------------------------------

_RE_REMOVED = re.compile(r"<removed>.*?</removed>", re.DOTALL)
_RE_ADDED = re.compile(r"<added>(.*?)</added>", re.DOTALL)
_RE_SUMMARY = re.compile(r"\*{0,2}\s*Summary of Changes\s*\*{0,2}.*", re.DOTALL | re.IGNORECASE)
_RE_CODE_FENCE = re.compile(r"^```[a-z]*\s*$", re.MULTILINE)
_RE_TRIPLE_QUOTE = re.compile(r'^"{3}\s*$', re.MULTILINE)
_RE_HORIZ_RULE = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_RE_PREFIX_LINE = re.compile(
    r"^(Revised DFARS Text|DFARS Text|Output|Revised Text)\s*:\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _resolve_draft(tagged: str) -> str:
    """Strip diff tags from a draft, keeping added text and removing deleted text."""
    text = _RE_REMOVED.sub("", tagged)
    text = _RE_ADDED.sub(r"\1", text)
    # Strip formatting artifacts
    text = _RE_SUMMARY.sub("", text)
    text = _RE_CODE_FENCE.sub("", text)
    text = _RE_TRIPLE_QUOTE.sub("", text)
    text = _RE_HORIZ_RULE.sub("", text)
    text = _RE_PREFIX_LINE.sub("", text)
    # Collapse runs of blank lines left by removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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


def _group_sections(changes: list[dict]) -> list[dict]:
    """Group changed nodes into section units with concatenated before/after text.

    Within a unit, node texts are concatenated in node-number order; empty sides
    (added nodes have no `before`, deleted nodes have no `after`) are skipped.
    """
    by_key: dict[str, list[dict]] = defaultdict(list)
    for ch in changes:
        by_key[_section_key(ch)].append(ch)

    sections: list[dict] = []
    for key, nodes in by_key.items():
        nodes = sorted(nodes, key=lambda c: c["number"])
        part = key.split(".")[0]
        before = "\n\n".join(n["before"].strip() for n in nodes if n.get("before", "").strip())
        after = "\n\n".join(n["after"].strip() for n in nodes if n.get("after", "").strip())
        if not before:
            # Pure additions (no prior text) -- nothing to draft "from"; skip.
            continue
        sections.append({
            "section": key,
            "part": part,
            "subpart": key.rsplit(".", 1)[0] if "." in key else part,
            "before": before,
            "after": after,
        })
    return sections


def load_ndaa_groups(single_ndaa: bool = False) -> list[dict]:
    """Build NDAA-centric groups from data/dfars_diff_all.json.

    Each diff entry pairs one NDAA (year, section) with the implementing DFARS
    case(s), and every case carries the before/after text of the DFARS nodes it
    changed (produced by dfars/dfars_diff.py). We pool the changed nodes across an
    NDAA's cases, roll them up to section units (see `_section_key`), and attach
    the pre-computed change manifest. NDAA sections without a manifest, or with no
    changed nodes carrying prior text, contribute nothing.

    Parameters
    ----------
    single_ndaa : bool
        If True, only keep DFARS sections that belong to NDAA (year, section)
        pairs with exactly one NDAA citation (fairer evaluation ground truth).

    Returns
    -------
    list of dict, each:
        {
          "ndaa": {"year": str, "section": str, "header": str, "text": str},
          "dfars_sections": [{"section": str, "part": str,
                              "subpart": str, "before": str, "after": str}, ...]
        }
    """
    allowed = get_single_ndaa_allowed() if single_ndaa else None

    with open(_DIFF_FILE) as f:
        diff: dict = json.load(f)

    # Pre-computed change manifests (from pipeline/fetch_context.py), keyed by
    # "<year>_<section>". These replace runtime manifest extraction.
    with open(_MANIFEST_FILE) as f:
        manifests = {s["ndaa_id"]: s for s in json.load(f).get("sections", [])}

    groups: list[dict] = []
    for entry in diff.get("sections", []):
        year = str(entry["ndaa_year"])
        section = str(entry["ndaa_section"])
        ndaa_id = f"{year}_{section}"

        manifest_entry = manifests.get(ndaa_id)
        if manifest_entry is None:
            print(f"  skip NDAA {year} s{section}: no pre-computed manifest")
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

        # The manifest list is injected verbatim as the change_manifest state;
        # delegation/drafting read it as embedded JSON text.
        change_manifest = json.dumps(
            manifest_entry.get("manifests", []), ensure_ascii=False
        )

        # Split into batches of 5 (rate-limit safety on the parallel fan-out)
        for batch_start in range(0, len(dfars_secs), 5):
            batch = dfars_secs[batch_start:batch_start + 5]
            groups.append({
                "ndaa": {
                    "year": year,
                    "section": section,
                    "header": manifest_entry.get("section_heading", ""),
                },
                "change_manifest": change_manifest,
                "dfars_sections": batch,
            })

    return groups


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    # -- inputs --
    ndaa_header: str
    dfars_sections: list[dict]  # [{section, part, subpart, before, after}]
    change_manifest: str   # JSON -- pre-computed manifest injected by the loader

    # -- phase 1: analysis --
    delegation_plan: str   # JSON

    # -- phase 2 (fan-out via Send, merged via operator.add) --
    section_drafts: Annotated[list[dict], operator.add]  # [{section, role, proposed_changes, draft}]

    # -- phase 3 --
    final_output: str


# ===================================================================
# Phase 1 -- Delegation
# ===================================================================


def delegation_agent(state: PipelineState) -> dict:
    """Assign manifest changes to specific DFARS sections."""
    llm = _get_llm().with_structured_output(DelegationPlan)

    summaries = "\n".join(
        f"  [{i}] {s['section']}  (Part: {s['part']}, Subpart: {s['subpart']})\n"
        f"      Full text:\n{s['before']}"
        for i, s in enumerate(state["dfars_sections"])
    )

    prompt = f"""You are a DFARS rulemaking coordinator.

Given the change manifest and the DFARS sections listed below, produce an
assignment plan that tells the drafting agents what each section needs.

CHANGE MANIFEST:
{state["change_manifest"]}

DFARS SECTIONS:
{summaries}

Roles:
- "primary": The section where the change is directly introduced — new definitions,
  new requirements, modified thresholds, or repeals.
- "secondary": The section is not the origin of the change but must adopt it —
  e.g., start using a new term defined in a primary section, apply a new threshold,
  or reflect a requirement introduced elsewhere.
- "cite-only": The section just needs to add or update a cross-reference to a
  primary or secondary section, with no substantive text changes.
- "unaffected": The NDAA has no impact on this section.

Rules:
- Specify the change_type: substantive, definitional, cross-reference, or none.
- Multiple sections can be "primary".
- It is possible that none of the sections listed here are "primary" — the primary
  section may not be in this batch. In that case, all sections can be "secondary",
  "cite-only", or "unaffected". Assign roles based on what each section needs,
  not on forcing a primary.
- Only ONE section should host new definitions unless they naturally belong apart.
- Mark sections that need no change as "unaffected".
- List which changes (by id) are assigned to each section.
- Provide specific delegation_notes for the drafter. For secondary sections,
  specify which primary section's changes they need to adopt.
"""
    result: DelegationPlan = llm.invoke([
        SystemMessage(
            content="You are a regulatory coordination expert."
        ),
        HumanMessage(content=prompt),
    ])
    return {"delegation_plan": result.model_dump_json()}


# ===================================================================
# Phase 2 -- Per-Section Drafting  (parallel via Send)
# ===================================================================


def route_to_sections(state: PipelineState) -> list[Send]:
    """Fan-out: create one Send per DFARS section for parallel drafting."""
    plan_raw = state["delegation_plan"]

    try:
        plan_data = json.loads(plan_raw)
        assignments = {
            a["section_index"]: a for a in plan_data.get("assignments", [])
        }
    except (json.JSONDecodeError, KeyError):
        assignments = {}

    sends: list[Send] = []
    for i, section in enumerate(state["dfars_sections"]):
        sends.append(Send("draft_single_section", {
            "section": section,
            "section_index": i,
            "assignment": assignments.get(i, {}),
            "change_manifest": state["change_manifest"],
            "ndaa_header": state.get("ndaa_header", ""),
        }))

    return sends


def draft_single_section(state: dict) -> dict:
    """Draft changes for a single DFARS section (runs as parallel subagent)."""
    section = state["section"]
    assign = state.get("assignment", {})
    manifest = state["change_manifest"]
    role = assign.get("role", "primary")

    # Skip unaffected sections -- no LLM calls
    if role == "unaffected":
        return {"section_drafts": [{
            "section": section["section"],
            "role": "unaffected",
            "proposed_changes": "No changes needed.",
            "draft": section["before"],
            "draft_clean": section["before"],
        }]}

    llm_cl = _get_llm(temperature=0.0)
    llm_draft = _get_llm(temperature=0.2)

    delegation_ctx = f"""
DELEGATION CONTEXT:
- Role: {role}
- Change type: {assign.get('change_type', 'substantive')}
- Hosts definitions: {assign.get('hosts_definitions', False)}
- Cites definitions from: {assign.get('cites_definitions_from', 'N/A')}
- Assigned changes: {assign.get('assigned_changes', [])}
- Notes: {assign.get('delegation_notes', '')}
"""

    # Role-specific guidance for change list and drafting
    role_guidance = ""
    if role == "cite-only":
        role_guidance = """
ROLE-SPECIFIC GUIDANCE (cite-only):
This section needs a NEW cross-reference paragraph inserted. Your task:
1. Determine the correct insertion point (follow existing numbering patterns).
2. Draft the cross-reference following the pattern of existing entries in this
   section (e.g., "Part XXX—[Title]. Use the [provision/clause] at 252.XXX-XXXX,
   [Title], as prescribed at [section], to comply with [statute].").
3. Note any paragraph renumbering required.
Do NOT conclude "no changes needed" — inserting the cross-reference IS the change.
"""
    elif role == "secondary":
        role_guidance = """
ROLE-SPECIFIC GUIDANCE (secondary):
This section must adopt changes from a primary section. Your task:
1. Identify text that references or depends on content being changed elsewhere.
2. Propose minimal updates for consistency (updated terms, thresholds, dates,
   or references).
3. Do NOT re-implement the primary change here.
"""

    draft_role_guidance = ""
    if role == "cite-only":
        draft_role_guidance = """
NOTE: This is a cite-only section. The main change is inserting a new
cross-reference paragraph. Follow the formatting pattern of existing entries.
Apply paragraph renumbering with <removed>/<added> tags where needed.
"""

    # ---- Step 1: Change list ----
    cl_prompt = f"""You are a DFARS rulemaking analyst.

Given the change manifest, delegation context, and DFARS section text below,
produce a **numbered list of specific changes** needed for this section.

CHANGE MANIFEST:
{manifest}
{delegation_ctx}

EXISTING DFARS TEXT:
\"\"\"
{section['before']}
\"\"\"
{role_guidance}
For each change specify:
  1. **Location** -- which paragraph/definition is affected
  2. **Type** -- ADD, REMOVE, or MODIFY
  3. **Description** -- what to change and why

MINIMALITY PRINCIPLE:
- Only propose changes DIRECTLY mandated by the NDAA and change manifest.
- Do NOT rewrite or rephrase existing text that remains legally valid.
- If the NDAA adds a new requirement, propose ADDING text — do not replace
  existing text unless it is directly contradicted.
- Prefer inserting sentences or paragraphs over replacing entire sections.

If no changes are needed, state that clearly.
"""
    cl_resp = llm_cl.invoke([
        SystemMessage(
            content=(
                "Be specific, actionable, and MINIMAL. Only list the smallest "
                "targeted changes necessary. Preserve all existing text that is "
                "not directly contradicted by the NDAA. Do not draft — only list changes."
            )
        ),
        HumanMessage(content=cl_prompt),
    ])
    proposed_changes = cl_resp.content.strip()

    # ---- Step 2: Draft ----
    draft_prompt = f"""You are an expert in DFARS regulations.

Implement the proposed changes into the DFARS text below.

PROPOSED CHANGES:
{proposed_changes}
{delegation_ctx}
{draft_role_guidance}
EXISTING DFARS TEXT:
\"\"\"
{section['before']}
\"\"\"

Instructions:
- Your goal is to make the MINIMUM changes necessary. Preserve all existing
  text verbatim unless a specific proposed change requires modifying it.
- Output the COMPLETE section text with changes applied inline.
- Wrap ONLY genuinely new text with <added>...</added> tags.
- Wrap ONLY genuinely removed text with <removed>...</removed> tags.
- For replacements: <removed>old</removed> <added>new</added>.
- Do NOT rephrase, reformat, or reorder existing text that is not changing.
- Do NOT include markdown, code fences, triple quotes, or commentary.
- Output ONLY the regulatory text with inline diff tags. No summaries,
  no preamble, no "Revised DFARS Text:" prefix.
"""
    draft_resp = llm_draft.invoke([
        SystemMessage(
            content=(
                "You are a DFARS drafter. Your paramount rule is MINIMAL EDITING: "
                "preserve every word of existing text that is not directly contradicted. "
                "Only add, remove, or modify the specific text required by the proposed changes. "
                "Output raw DFARS regulatory text only — no markdown, no commentary, no summaries.\n\n"
                "Follow the FAR/DFARS Drafting Guide conventions below:\n\n" + _DRAFTING_GUIDE
            )
        ),
        HumanMessage(content=draft_prompt),
    ])

    draft_tagged = draft_resp.content.strip()
    return {"section_drafts": [{
        "section": section["section"],
        "role": role,
        "proposed_changes": proposed_changes,
        "draft": draft_tagged,
        "draft_clean": _resolve_draft(draft_tagged),
    }]}


# ===================================================================
# Phase 3 -- Reconciliation
# ===================================================================


def reconciliation_agent(state: PipelineState) -> dict:
    """Review all section drafts together for consistency."""
    llm = _get_llm(temperature=0.0)

    drafts_text = "\n\n".join(
        f"=== {d['section']} (role: {d.get('role', '?')}) ===\n{d['draft']}"
        for d in state.get("section_drafts", [])
    )

    prompt = f"""You are an expert in reviewing DFARS drafts.

You have draft revisions for multiple DFARS sections, all implementing the same NDAA provision. Review them together and check:

1. **Definition dedup** -- definitions belong only in the host section; other sections should cross-reference, not duplicate.
2. **Cross-reference validation** -- cite-only sections actually cite the right source.
3. **Consistency** -- terms, thresholds, dates identical across sections.
4. **Conflict resolution** -- no two sections implement the same sub-requirement differently.

CHANGE MANIFEST:
{state["change_manifest"]}

DELEGATION PLAN:
{state["delegation_plan"]}

DRAFT REVISIONS:
{drafts_text}

Output:
1. Issues found (if any).
2. For each section, the final revised text (or "no changes from draft").
3. A brief overall summary.
"""
    resp = llm.invoke([
        SystemMessage(
            content="Be thorough but concise. Focus on cross-section consistency."
        ),
        HumanMessage(content=prompt),
    ])
    return {"final_output": resp.content.strip()}


# ===================================================================
# Graph assembly
# ===================================================================


def build_graph():
    """Construct and compile the 1:N pipeline graph."""
    wf = StateGraph(PipelineState)

    # Phase 1
    wf.add_node("delegation", delegation_agent)

    # Phase 2 -- parallel fan-out via Send()
    wf.add_node("draft_single_section", draft_single_section)

    # Phase 3
    wf.add_node("reconciliation", reconciliation_agent)

    # Edges
    wf.set_entry_point("delegation")
    # Fan-out: delegation -> N parallel draft_single_section instances
    wf.add_conditional_edges("delegation", route_to_sections)
    # Fan-in: each draft_single_section merges into section_drafts via reducer
    wf.add_edge("draft_single_section", "reconciliation")
    wf.add_edge("reconciliation", END)

    return wf.compile()


# ===================================================================
# Public helper
# ===================================================================


def run_pipeline(group: dict) -> PipelineState:
    """Run the full 1:N pipeline for a single NDAA group."""
    graph = build_graph()
    initial: PipelineState = {
        "ndaa_header": group["ndaa"].get("header", ""),
        "dfars_sections": group["dfars_sections"],
        "change_manifest": group["change_manifest"],
        "delegation_plan": "",
        "section_drafts": [],
        "final_output": "",
    }
    return graph.invoke(initial)


# ===================================================================
# CLI runner
# ===================================================================

if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv(_FRAMEWORK_DIR / ".env")

    parser = argparse.ArgumentParser(description="1:N NDAA -> DFARS Pipeline")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max NDAA groups to process"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Process NDAA groups in batches of this size (rate-limit safety)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--single-ndaa", action="store_true",
        help="Only process DFARS sections from documents with exactly one NDAA citation",
    )
    args = parser.parse_args()

    print("Loading NDAA -> DFARS groups ...")
    groups = load_ndaa_groups(single_ndaa=args.single_ndaa)
    print(f"Found {len(groups)} NDAA groups")

    if args.limit:
        groups = groups[: args.limit]

    # Process in batches to respect rate limits
    print(f"Processing in batches of {args.batch_size}")

    results: list[dict] = []
    for idx, group in enumerate(groups, 1):
        ndaa = group["ndaa"]
        n = len(group["dfars_sections"])
        print(
            f"\n[{idx}/{len(groups)}] NDAA {ndaa['year']} "
            f"s{ndaa['section']} -> {n} DFARS section(s)"
        )

        try:
            result = run_pipeline(group)
            # Attach ground-truth "before"/"after" text to each section draft
            gt_by_section = {
                s["section"]: s
                for s in group["dfars_sections"]
            }
            drafts = result.get("section_drafts", [])
            for d in drafts:
                gt = gt_by_section.get(d["section"], {})
                d["before"] = gt.get("before", "")
                d["after"] = gt.get("after", "")
            results.append({
                "ndaa_year": ndaa["year"],
                "ndaa_section": ndaa["section"],
                "ndaa_header": ndaa.get("header", ""),
                "n_dfars_sections": n,
                "section_drafts": drafts,
                "final_output": result.get("final_output", ""),
            })
            print(f"  done -- {len(result.get('section_drafts', []))} sections drafted")
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({
                "ndaa_year": ndaa["year"],
                "ndaa_section": ndaa["section"],
                "error": str(exc),
            })

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(_DATA_DIR / "results" / f"pipeline_1n_results_{ts}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
