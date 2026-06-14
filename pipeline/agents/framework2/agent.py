"""
1:N NDAA -> DFARS Drafting Pipeline
====================================
Three-phase LangGraph pipeline that processes one NDAA section against
N DFARS sections simultaneously:

  Phase 1 -- Delegation (once per NDAA group)
    delegation: route each numbered change to the DFARS node(s) that implement it

  Phase 2 -- Per-Node Drafting (PARALLEL per DFARS node via Send())
    Each node implements only the changes routed to it -- one LLM call

  Phase 3 -- Reconciliation (mode-selectable; see build_graph)
    "off"         -- skip; per-node drafts are final
    "per_section" -- coordinator routes corrections, applied per section in parallel

Inputs:
  - data/dfars_diff_all.json: per NDAA (year, section), the implementing DFARS
    case(s) and the before/after text of every changed DFARS node (produced by
    dfars/dfars_diff.py from dated eCFR snapshots in Mongo). Changed nodes are
    rolled up to their enclosing SECTION (part-252 clauses kept whole) so each
    drafting unit carries full-section context.
  - pipeline/out/manifests_fr_cases.json: the pre-computed change manifest per NDAA
    section (from pipeline/fetch_context.py). Sections without one are skipped.
    Surfaced to the prompts as a compact change list (change_type, description,
    applies_to only).
  - Mongo (db "ndaa_dfars", collection "ndaas"): the NDAA section's full statutory
    text, fed to the prompts alongside the change list. Read-only.
"""

from __future__ import annotations

import json
import operator
import os
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

from agents.framework2.schemas import (  # noqa: E402
    DelegationPlan,
    ReconcilePlan,
    SectionDraft,
)
from agents.framework2.utils import _section_key, get_single_ndaa_allowed  # noqa: E402
from utils.mongo_utils import getMongoClient, get_doc_by_year_section  # noqa: E402

DB = "ndaa_dfars"
NDAAS = "ndaas"

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


def _get_llm(temperature: float = 0.0, max_tokens: int = 4096) -> AzureChatOpenAI:
    # The 4096 default suits compact outputs (e.g. the delegation routing plan,
    # which is just node->change-number lists). Calls that emit full regulatory
    # text pass a larger budget explicitly: per-section drafting returns one
    # section's complete revised text (some exceed 4096 -> truncation fails
    # structured-output parsing and drops the whole group), and reconciliation
    # returns the full text of every section in one call.
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
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _render_change(i: int, m: dict) -> str:
    """Render a single manifest change, numbered `i`."""
    return (
        f"{i}. [{m.get('change_type', '')}] {m.get('description', '')}\n"
        f"   Applies to: {m.get('applies_to', '')}"
    )


def _change_items(manifests: list[dict]) -> list[str]:
    """One rendered change string per manifest item, 1-indexed in the text.

    `_change_items(...)[k]` is the change shown as number `k + 1` in the change
    list -- delegation routes by these numbers, and route_to_sections resolves a
    node's assigned numbers back to these strings.
    """
    return [_render_change(i, m) for i, m in enumerate(manifests, 1)]


def _format_change_list(manifests: list[dict]) -> str:
    """Render the manifest as a compact, numbered change list.

    Only the fields the drafting stages need are surfaced: change_type,
    description, and applies_to.
    """
    items = _change_items(manifests)
    return "\n".join(items) if items else "(no changes inferred)"


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

    client = getMongoClient()
    groups: list[dict] = []
    try:
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

            # Full NDAA statutory text (read-only from Mongo) feeds the prompts
            # alongside the compact change list.
            ndaa_doc = get_doc_by_year_section(client, DB, NDAAS, year, section)
            if ndaa_doc is None:
                print(f"  skip NDAA {year} s{section}: not found in Mongo '{NDAAS}'")
                continue
            ndaa_section = ndaa_doc.get("section", {})

            # Compact change list: change_type, description, applies_to only.
            # `change_items` keeps the per-change strings addressable by number so
            # delegation can route them and drafting can resolve a node's subset.
            manifest_items = manifest_entry.get("manifests", [])
            change_list = _format_change_list(manifest_items)
            change_items = _change_items(manifest_items)

            # One group per NDAA: all affected DFARS sections stay together so
            # delegation sees the full set and drafting fans out over every node.
            groups.append({
                "ndaa": {
                    "year": year,
                    "section": section,
                    "header": ndaa_section.get("heading", manifest_entry.get("section_heading", "")),
                    "text": ndaa_section.get("text", ""),
                },
                "change_list": change_list,
                "change_items": change_items,
                "dfars_sections": dfars_secs,
            })
    finally:
        client.close()

    return groups


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    # -- inputs --
    ndaa_header: str
    ndaa_text: str         # full NDAA statutory text (from Mongo)
    dfars_sections: list[dict]  # [{section, part, subpart, before, after}]
    change_list: str       # compact numbered change list (type/description/applies_to)
    change_items: list[str]  # per-change strings, addressable by change number

    # -- phase 1: delegation (change -> node routing) --
    delegation_plan: str   # JSON: {assignments: [{node_index, change_numbers}]}

    # -- phase 2 (fan-out via Send, merged via operator.add) --
    section_drafts: Annotated[list[dict], operator.add]  # [{section, assigned_changes, draft_clean}]

    # -- phase 3 --
    # Reconciled per-section drafts (section_drafts with draft_clean overwritten
    # by the cross-section-consistent text). Kept separate from section_drafts so
    # it last-write-wins instead of appending through the operator.add reducer.
    reconciled_drafts: list[dict]
    final_output: str       # human-readable summary of reconciliation issues

    # -- phase 3 (per-section mode only) --
    reconcile_directive: str  # JSON ReconcilePlan: {corrections:[{section, corrections}], issues}
    # Fan-in of the per-section apply step; assembled into reconciled_drafts.
    reconciled_parts: Annotated[list[dict], operator.add]


# ===================================================================
# Phase 1 -- Delegation
# ===================================================================


def delegation_agent(state: PipelineState) -> dict:
    """Route each numbered change to the DFARS node(s) that must implement it.

    This is delegation's only job: produce a change -> node mapping. It performs
    no drafting and assigns no roles. Each node then receives exactly the subset
    of changes routed to it.
    """
    llm = _get_llm().with_structured_output(DelegationPlan)

    nodes = "\n".join(
        f"  [{i}] {s['section']}  (Part: {s['part']}, Subpart: {s['subpart']})\n"
        f"      Current text:\n{s['before']}"
        for i, s in enumerate(state["dfars_sections"])
    )

    prompt = f"""You are a DFARS rulemaking coordinator. Your ONLY job is to route
changes to the DFARS nodes that must implement them. Do not draft anything.

Following is the NDAA section:
<NDAA_SECTION>
{state["ndaa_text"]}
</NDAA_SECTION>

NUMBERED CHANGES (inferred from the cited sources):
{state["change_list"]}

DFARS NODES (indexed):
{nodes}

For each numbered change above, decide which DFARS node(s) must implement it. A
change may map to one node, to several nodes, or -- if no listed node is the
right place for it -- to none. A node may receive several changes or none.

Return, for EACH node, the list of change numbers it must implement (by their
number in the change list above), using the node index exactly as shown. Map a
change to a node only where that node's own text is what actually has to change.
"""
    result: DelegationPlan = llm.invoke([
        SystemMessage(
            content="You are a regulatory coordination expert. Map changes to "
            "nodes -- do not draft."
        ),
        HumanMessage(content=prompt),
    ])
    return {"delegation_plan": result.model_dump_json()}


# ===================================================================
# Phase 2 -- Per-Node Drafting  (parallel via Send)
# ===================================================================


def route_to_sections(state: PipelineState) -> list[Send]:
    """Fan-out: one Send per DFARS node, carrying the changes routed to it.

    Resolves each node's assigned change numbers back to their rendered change
    strings so the drafter receives only its own subset of changes.
    """
    try:
        plan_data = json.loads(state["delegation_plan"])
        by_node = {
            a["node_index"]: a.get("change_numbers", [])
            for a in plan_data.get("assignments", [])
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        by_node = {}

    change_items = state.get("change_items", [])

    sends: list[Send] = []
    for i, section in enumerate(state["dfars_sections"]):
        nums = by_node.get(i, [])
        assigned = [
            change_items[n - 1]
            for n in nums
            if isinstance(n, int) and 1 <= n <= len(change_items)
        ]
        sends.append(Send("draft_single_section", {
            "section": section,
            "assigned_changes": assigned,
            "ndaa_text": state["ndaa_text"],
        }))

    return sends


def draft_single_section(state: dict) -> dict:
    """Draft a single DFARS node by implementing the changes routed to it.

    Runs as a parallel subagent (one per node). Delegation has already decided
    which numbered changes this node implements, so this is a single LLM call:
    apply those changes to the node text. A node with no changes routed to it is
    emitted unchanged -- no LLM call.
    """
    section = state["section"]
    assigned = state.get("assigned_changes", [])
    ndaa_text = state["ndaa_text"]

    # No change routed here -> nothing to implement; emit current text as-is.
    if not assigned:
        return {"section_drafts": [{
            "section": section["section"],
            "assigned_changes": [],
            "draft_clean": section["before"],
        }]}

    changes_block = "\n\n".join(assigned)

    draft_prompt = f"""You are an expert in DFARS regulations. Implement the changes
assigned to this section into its text.

Following is the NDAA section that mandates these changes:
<NDAA_SECTION>
{ndaa_text}
</NDAA_SECTION>

CHANGES TO IMPLEMENT IN THIS SECTION:
{changes_block}

EXISTING DFARS TEXT:
\"\"\"
{section['before']}
\"\"\"

Instructions:
- Implement EVERY change listed above, and ONLY those changes.
- Make the MINIMUM edits necessary. Preserve all existing text verbatim unless a
  listed change requires modifying it.
- Return the COMPLETE revised section text with the changes applied inline.
- Do NOT rephrase, reformat, or reorder existing text that is not changing.
"""
    llm = _get_llm(temperature=0.0, max_tokens=16000).with_structured_output(SectionDraft)
    result: SectionDraft = llm.invoke([
        SystemMessage(
            content=(
                "You are a DFARS drafter. Your paramount rule is MINIMAL EDITING: "
                "preserve every word of existing text that is not directly contradicted. "
                "Implement only the changes assigned to this section. "
                "Return the full revised section text only.\n\n"
                "Follow the FAR/DFARS Drafting Guide conventions below:\n\n" + _DRAFTING_GUIDE
            )
        ),
        HumanMessage(content=draft_prompt),
    ])

    return {"section_drafts": [{
        "section": section["section"],
        "assigned_changes": assigned,
        "draft_clean": result.revised_text.strip(),
    }]}


# ===================================================================
# Phase 3 -- Reconciliation (plan corrections, then apply per section)
# ===================================================================


def reconcile_plan(state: PipelineState) -> dict:
    """Coordinator: review all drafts together and route per-section corrections.

    Mirrors delegation's shape -- a single global call that emits per-node
    assignments -- but here it both discovers the cross-section issues and routes
    the specific corrections each section needs. It drafts nothing; a separate
    per-section step applies the corrections (bounding each output to one
    section, so large groups can't truncate the way the joint call can).
    """
    # Output is corrections + a summary, not full section text, so a moderate
    # budget suffices even for large groups.
    llm = _get_llm(temperature=0.0, max_tokens=8000).with_structured_output(
        ReconcilePlan
    )

    drafts = state.get("section_drafts", [])
    drafts_text = "\n\n".join(
        f"=== {d['section']} ===\n{d['draft_clean']}"
        for d in drafts
    )

    prompt = f"""You are an expert reviewing DFARS drafts. You have draft revisions
for multiple DFARS sections, all implementing the same NDAA provision. Review them
together and identify cross-section inconsistencies -- but do NOT rewrite the
sections. Only list the specific corrections each section needs.

Check for:
1. **Definition dedup** -- definitions belong only in the host section; other sections should cross-reference, not duplicate.
2. **Cross-reference validation** -- sections cite the right source section.
3. **Consistency** -- terms, thresholds, dates identical across sections.
4. **Conflict resolution** -- no two sections implement the same sub-requirement differently.

Below are the changes inferred by looking at the external sources cited:
{state["change_list"]}

DRAFT REVISIONS:
{drafts_text}

For each section that needs changes, return its section number (exactly as shown)
and a list of specific, minimal corrections to apply to its draft. A section that
is already consistent needs NO entry. Also give a brief summary of the issues you
found.
"""
    result: ReconcilePlan = llm.invoke([
        SystemMessage(
            content="Be thorough but concise. Identify cross-section issues and "
            "route corrections -- do not draft."
        ),
        HumanMessage(content=prompt),
    ])
    return {
        "reconcile_directive": result.model_dump_json(),
        "final_output": result.issues.strip(),
    }


def route_reconcile(state: PipelineState) -> list[Send]:
    """Fan-out: one Send per draft, carrying that section's corrections."""
    try:
        plan = json.loads(state["reconcile_directive"])
        by_section = {
            c["section"]: c.get("corrections", [])
            for c in plan.get("corrections", [])
        }
        issues = plan.get("issues", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        by_section, issues = {}, ""

    sends: list[Send] = []
    for d in state.get("section_drafts", []):
        sends.append(Send("reconcile_single_section", {
            "draft": d,
            "corrections": by_section.get(d["section"], []),
            "review_context": issues,
        }))
    return sends


def reconcile_single_section(state: dict) -> dict:
    """Apply one section's routed corrections to its draft (parallel subagent).

    A section with no corrections passes through unchanged -- no LLM call.
    """
    draft = state["draft"]
    corrections = state.get("corrections", [])

    nd = dict(draft)
    if not corrections:
        return {"reconciled_parts": [nd]}

    corrections_block = "\n".join(f"- {c}" for c in corrections)
    prompt = f"""You are reconciling one DFARS section's draft for cross-section
consistency with the other sections implementing the same NDAA provision.

REVIEW CONTEXT (cross-section issues found across all sections):
{state.get('review_context', '') or '(none provided)'}

CORRECTIONS TO APPLY TO THIS SECTION:
{corrections_block}

CURRENT DRAFT (section {draft['section']}):
\"\"\"
{draft['draft_clean']}
\"\"\"

Instructions:
- Apply ONLY the corrections listed above. Make the MINIMUM edits necessary.
- Preserve all other existing text verbatim.
- Return the COMPLETE revised section text.
"""
    llm = _get_llm(temperature=0.0, max_tokens=8000).with_structured_output(
        SectionDraft
    )
    result: SectionDraft = llm.invoke([
        SystemMessage(
            content=(
                "You are a DFARS drafter applying targeted cross-section "
                "corrections. Minimal editing: change only what the corrections "
                "require, and preserve every other word verbatim. Return the full "
                "revised section text only.\n\n"
                "Follow the FAR/DFARS Drafting Guide conventions below:\n\n" + _DRAFTING_GUIDE
            )
        ),
        HumanMessage(content=prompt),
    ])

    revised = result.revised_text.strip()
    if revised:
        nd["draft_clean"] = revised
    return {"reconciled_parts": [nd]}


def assemble_reconciled(state: PipelineState) -> dict:
    """Fan-in: collect the per-section applied drafts into reconciled_drafts."""
    return {"reconciled_drafts": state.get("reconciled_parts", [])}


# ===================================================================
# Graph assembly
# ===================================================================


RECONCILE_MODES = ("off", "per_section")


def build_graph(reconcile: str = "per_section"):
    """Construct and compile the 1:N pipeline graph.

    ``reconcile`` selects the Phase-3 cross-section reconciliation:

    - "off"          -- skip it; the per-node drafts are final. Isolates whether
                        decomposition alone (route + isolated drafting) helps.
    - "per_section"  -- a coordinator routes per-section corrections, then a
                        parallel step applies each section's corrections (one
                        bounded output per call; no large-group truncation).
    """
    if reconcile not in RECONCILE_MODES:
        raise ValueError(
            f"reconcile must be one of {RECONCILE_MODES}, got {reconcile!r}"
        )

    wf = StateGraph(PipelineState)

    # Phase 1 -- delegation (change -> node routing)
    wf.add_node("delegation", delegation_agent)

    # Phase 2 -- parallel fan-out via Send()
    wf.add_node("draft_single_section", draft_single_section)

    wf.set_entry_point("delegation")
    # Fan-out: delegation -> N parallel draft_single_section instances
    wf.add_conditional_edges("delegation", route_to_sections)

    if reconcile == "per_section":
        # Phase 3 -- plan corrections (1 call), then apply per section (fan-out)
        wf.add_node("reconcile_plan", reconcile_plan)
        wf.add_node("reconcile_single_section", reconcile_single_section)
        wf.add_node("assemble_reconciled", assemble_reconciled)
        wf.add_edge("draft_single_section", "reconcile_plan")
        wf.add_conditional_edges("reconcile_plan", route_reconcile)
        wf.add_edge("reconcile_single_section", "assemble_reconciled")
        wf.add_edge("assemble_reconciled", END)
    else:  # "off"
        wf.add_edge("draft_single_section", END)

    return wf.compile()


# ===================================================================
# Public helper
# ===================================================================


def run_pipeline(group: dict, reconcile: str = "per_section") -> PipelineState:
    """Run the full 1:N pipeline for a single NDAA group.

    ``reconcile`` is the Phase-3 mode: "off" or "per_section" (see build_graph).
    """
    graph = build_graph(reconcile=reconcile)
    initial: PipelineState = {
        "ndaa_header": group["ndaa"].get("header", ""),
        "ndaa_text": group["ndaa"].get("text", ""),
        "dfars_sections": group["dfars_sections"],
        "change_list": group["change_list"],
        "change_items": group.get("change_items", []),
        "delegation_plan": "",
        "section_drafts": [],
        "reconciled_drafts": [],
        "final_output": "",
        "reconcile_directive": "",
        "reconciled_parts": [],
    }
    return graph.invoke(initial)


# ===================================================================
# Pre-compiled graphs for LangGraph Studio / `langgraph dev`
# ===================================================================
# The dev server treats a langgraph.json factory as `make_graph(config)` and
# passes it the RunnableConfig -- which would collide with build_graph's
# `reconcile` arg. So expose ready-made compiled graphs and point langgraph.json
# at these instead of at build_graph.
graph = build_graph("per_section")   # default Studio graph
graph_off = build_graph("off")       # reconciliation-off variant


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
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--single-ndaa", action="store_true",
        help="Only process DFARS sections from documents with exactly one NDAA citation",
    )
    parser.add_argument(
        "--reconcile", choices=["off", "per-section"], default="per-section",
        help="Phase-3 reconciliation mode (default: per-section). 'off' = per-node "
        "drafts are final; 'per-section' = plan corrections then apply per section",
    )
    args = parser.parse_args()
    reconcile_mode = args.reconcile.replace("-", "_")

    print("Loading NDAA -> DFARS groups ...")
    groups = load_ndaa_groups(single_ndaa=args.single_ndaa)
    print(f"Found {len(groups)} NDAA groups")

    if args.limit:
        groups = groups[: args.limit]

    results: list[dict] = []
    for idx, group in enumerate(groups, 1):
        ndaa = group["ndaa"]
        n = len(group["dfars_sections"])
        print(
            f"\n[{idx}/{len(groups)}] NDAA {ndaa['year']} "
            f"s{ndaa['section']} -> {n} DFARS section(s)"
        )

        try:
            result = run_pipeline(group, reconcile=reconcile_mode)
            # Attach ground-truth "before"/"after" text to each section draft.
            # Prefer the reconciled drafts (draft_clean overwritten with the
            # cross-section-consistent text); fall back to the raw Phase-2
            # drafts if reconciliation produced nothing.
            gt_by_section = {
                s["section"]: s
                for s in group["dfars_sections"]
            }
            drafts = result.get("reconciled_drafts") or result.get("section_drafts", [])
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
