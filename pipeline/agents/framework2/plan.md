# 1:N NDAA → DFARS Pipeline Design

## Problem

The current pipeline is DFARS-centric: each `valid_pairs.json` entry is `{dfars: {...}, ndaas: [...]}`, and `main.py` flattens this into independent 1:1 pairs. Each pair is processed in complete isolation — the graph has no knowledge that the same NDAA is being drafted against multiple DFARS sections simultaneously.

This means:
- Definitions get duplicated across every impacted section
- No section can "cite" another section's implementation because it doesn't know about it
- There's no coordination on which section is the natural home for a substantive change
- The full NDAA text is repeated at every agent call × every section — the biggest token multiplier

## Architecture: Three-Phase Pipeline

```
Phase 1: NDAA Analysis          Phase 2: Per-Section Drafting      Phase 3: Reconciliation
(runs ONCE per NDAA)            (runs per DFARS section)           (runs ONCE per NDAA group)

NDAA full text                  Per section:                       All N drafts
     │                          assigned changes + DFARS text           │
     ▼                               │                                 ▼
[Citation Extractor]                 ▼                          [Reconciliation Agent]
     │                          [Change List Agent]              - Dedup definitions
     ▼                               │                           - Cross-ref validation
[Web Search Researcher]              ▼                           - Consistency check
     │  (loop if needed)        [Drafting Agent]                 - Conflict resolution
     ▼                               │                                 │
[Change Extraction Agent]            ▼                                 ▼
     │                          Section draft                    Final revised drafts
     ▼
Change Manifest (compact)
     │
     ▼
[Delegation Agent]
  manifest + N DFARS identifiers
     │
     ▼
Assignment plan per section
```

## Phase 1 — NDAA Analysis (runs once, no DFARS context)

### Citation Extractor + Web Search Loop

The existing citation extraction and web search loop moves here. It operates on the NDAA alone since the citations it resolves (USC sections, public laws, etc.) are properties of the NDAA, not of any specific DFARS section.

- **Citation Extractor**: Pulls USC/PL/executive order references from the NDAA text
- **Web Search Researcher**: Resolves those citations via web search, summarizes findings
- Loops until citations are exhausted or max iterations reached

### Change Extraction Agent (NEW)

With enriched context from the research loop, produces a **change manifest** — a compact, structured representation of everything the NDAA mandates:

```json
{
  "mandated_changes": [
    {
      "id": "C1",
      "type": "new_requirement",
      "description": "Require third-party CMMC assessments for contractors handling CUI",
      "statutory_basis": "Sec 1505(b)",
      "key_terms": ["CMMC", "third-party assessment", "NIST SP 800-171"],
      "relevant_ndaa_subsection": "(b) ASSESSMENT REQUIREMENTS..."
    },
    {
      "id": "C2",
      "type": "modified_threshold",
      "description": "Cyber incident reporting deadline changed to 72 hours",
      "statutory_basis": "Sec 1505(c), 10 USC 391"
    }
  ],
  "definitions_introduced": ["Cybersecurity Maturity Model Certification"],
  "authorities_amended": ["10 USC 391", "48 CFR 204.73"],
  "effective_date": "180 days after enactment"
}
```

This manifest is ~200-400 tokens vs 2000-5000 for the raw NDAA text. Every downstream agent uses the manifest instead of the full NDAA.

### Delegation Agent (NEW)

**Input:** Change manifest + N DFARS section identifiers/summaries (not full text — just part, subpart, section name, brief description)

**Output:** An assignment plan. For each DFARS section:

| Field | Values | Purpose |
|---|---|---|
| `role` | `primary`, `secondary`, `cite-only`, `unaffected` | How much change this section needs |
| `change_type` | `substantive`, `definitional`, `cross-reference`, `none` | Nature of the change |
| `hosts_definitions` | `bool` | Whether this section should add/modify definitions |
| `cites_definitions_from` | `str \| null` | Which other DFARS section's definitions to reference |
| `assigned_changes` | `list[str]` | Change IDs from the manifest assigned to this section |
| `delegation_notes` | `str` | Specific instructions for the drafting agent |

Key: **multiple sections can be `primary`**. The delegation agent doesn't force a single primary — it can split substantive changes across sections when that's the natural fit.

Also includes a `cross_reference_map` — which sections should reference which other sections.

## Phase 2 — Per-Section Drafting (runs per DFARS section)

The per-section graph is now just **2 sequential calls** (no more research loop — that happened in Phase 1):

1. **Change List Agent** (existing, modified prompt) — receives:
   - Assigned changes from the manifest (not the full NDAA)
   - Delegation plan (role, whether to host definitions, cross-ref instructions)
   - The full DFARS section text (before)
   - Research context from Phase 1

2. **Drafting Agent** (existing, modified prompt) — receives:
   - The change list from above
   - Delegation context
   - The DFARS text
   - The FAR/DFARS Drafting Guide

The delegation context steers agents away from duplicating work:
- "You are drafting a `cite-only` change. Do NOT add definitions. Reference DFARS 252.204-7012 for definitions."
- "You are the `hosts_definitions` section. Add these definitions here."
- "Other sections being changed: 204.7300 (scope), 252.204-7020 (reporting). Your focus is the contractor clause."

These N section-level graphs can run in parallel via LangGraph `Send()`.

## Phase 3 — Reconciliation Agent (NEW, runs once per group)

**Input:** All N draft outputs + the delegation plan + change manifest

**Job:**
1. **Definition dedup** — if definitions leaked into a non-host section, remove them and add a cross-reference
2. **Cross-reference validation** — ensure every `cite-only` section actually cites the right source section
3. **Consistency check** — terms used identically across sections, no contradictory thresholds/dates
4. **Conflict resolution** — if two sections both claim to implement the same sub-requirement, reconcile

**Output:** Final revised drafts for all N sections.

## Token Cost Comparison

For 1 NDAA affecting 5 DFARS sections, research loop doing 2 iterations:

| | Current (1:1) | Proposed (1:N) |
|---|---|---|
| Research loop (citation + web search + relatedness) | 5 sections × ~3 calls × 2 iters = **30 LLM calls**, each with full NDAA | **~6 calls** total (one NDAA, looped twice) |
| Change list + drafting | 5 × 2 = **10 calls**, each with full NDAA | 5 × 2 = **10 calls**, but with compact manifest |
| Delegation | N/A | **1 call** |
| Reconciliation | N/A | **1 call** |
| **Total LLM calls** | **~40** | **~18** |
| **NDAA text copies sent** | ~40 | **1** (full text only in extraction phase) |

Roughly **50-60% fewer tokens** while producing coordinated, non-redundant output.

## Data Model Changes

### New file: `ndaa_groups.json`

Reverse index from `valid_pairs.json`, grouping by NDAA → list of DFARS sections:

```json
{
  "ndaa": {"year": "2022", "section": "845", "title": "...", "text": "..."},
  "dfars_sections": [
    {"section": "252.204-7012", "part": "252", "subpart": "252.204", "before": "...", "after": "..."},
    {"section": "204.7300", "part": "204", "subpart": "204.73", "before": "...", "after": "..."},
    {"section": "204.7301", "part": "204", "subpart": "204.73", "before": "...", "after": "..."}
  ]
}
```

### `get_mapping.py` addition

After existing `valid_pairs.json` generation, build NDAA-centric groups:

```python
from collections import defaultdict

ndaa_groups = defaultdict(lambda: {"ndaa": None, "dfars_sections": []})

for pair in pairs_export:
    for ndaa in pair["ndaas"]:
        key = (ndaa["year"], ndaa["section"])
        if ndaa_groups[key]["ndaa"] is None:
            ndaa_groups[key]["ndaa"] = ndaa
        ndaa_groups[key]["dfars_sections"].append(pair["dfars"])

ndaa_groups_export = [
    {"ndaa": g["ndaa"], "dfars_sections": g["dfars_sections"]}
    for g in ndaa_groups.values()
    if len(g["dfars_sections"]) > 0
]
# Save as ndaa_groups.json
```

## Graph Structure (LangGraph)

```python
# Outer graph (1:N, per NDAA group)
START
  → citation_extractor          # existing, but NDAA-only (no DFARS context)
  → web_search_researcher       # existing
  → route: loop back or continue
  → change_extraction_agent     # NEW: produces manifest from enriched NDAA
  → delegation_agent            # NEW: manifest + DFARS identifiers → assignment plan
  → fan_out (Send per section)
  │   → change_list_agent       # existing, receives manifest slice + DFARS text
  │   → drafting_agent          # existing, receives manifest slice + DFARS text
  → fan_in
  → reconciliation_agent        # NEW: reviews all drafts together
  → END
```

## Files to Create/Modify

| File | Change | Scope |
|---|---|---|
| `agents/graph_1n.py` | **NEW** — full 1:N graph with all phases | ~500 lines |
| `agents/main_1n.py` | **NEW** — CLI runner for 1:N mode | ~150 lines |
| `get_mapping.py` | Add `ndaa_groups.json` output (or separate script) | ~30 lines |
| `agents/eval.py` | Eventually add group-level scoring | Future |

Existing `graph.py` and `main.py` remain untouched for backward compatibility.

## Open Design Questions

1. **Parallelism in Phase 2** — LangGraph `Send()` can run all N section-level graphs in parallel. Faster but costs more concurrent tokens. Could be configurable.

2. **Delegation agent errors** — The reconciliation agent acts as a safety net. If a section tagged `unaffected` actually needs changes, reconciliation should catch it. A feedback loop from reconciliation back to delegation is possible but adds complexity — start without it.

3. **Manifest granularity** — How detailed should each change entry be? Too terse and the drafter loses nuance. Too verbose and we lose the token savings. Target: each change is 1-3 sentences + the relevant NDAA subsection snippet (not the full NDAA).
