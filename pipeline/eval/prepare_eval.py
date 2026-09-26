"""
Prepare a ground-truth change inventory for evaluating NDAA -> DFARS drafts.

For every DFARS section-unit amended by an NDAA provision, this distills the
official before -> after delta into a concise list of *logical* changes (e.g. a
threshold change, a citation update, an added definition). The result is keyed
by ``<ndaaYear>_<ndaaSection>`` and broken down per DFARS section-unit, using the
same section-unit keys the drafting pipeline emits, so the change lists can later
be matched against a run's section drafts for change-level evaluation.

Source data comes straight from ``data/dfars_diff_all.json`` (via
``expected_section_units`` with the single-NDAA and manifest filters off); every
NDAA section in the diff is covered (only pure-addition and >25-unit groups are
skipped).

Usage (from the repo root)
--------------------------
  # Build the full inventory (one LLM call per section-unit)
  python pipeline/eval/prepare_eval.py

  # Write somewhere specific
  python pipeline/eval/prepare_eval.py --output data/eval/section_changes.json

  # Quick test run over the first 2 NDAA sections
  python pipeline/eval/prepare_eval.py --limit 2

  # Spot-check a single NDAA section
  python pipeline/eval/prepare_eval.py --ndaa 2024_2881
"""

import argparse
import json
import os
import sys
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _EVAL_DIR.parents[0]        # pipeline/ -- for `agents.*` imports
_REPO_ROOT = _EVAL_DIR.parents[1]           # ndaa_dfars/ -- for data/
_FRAMEWORK_DIR = _PIPELINE_DIR / "agents" / "framework2"  # for .env
_DATA_DIR = _REPO_ROOT / "data"
sys.path.insert(0, str(_PIPELINE_DIR))

DEFAULT_OUTPUT = _DATA_DIR / "eval" / "section_changes.json"

from agents.framework2.utils import expected_section_units  # noqa: E402


# ---------------------------------------------------------------------------
# LLM
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
# Change-extraction schema (structured output)
# ---------------------------------------------------------------------------

class Change(BaseModel):
    """A single logical change made to a DFARS section-unit."""
    category: str = Field(
        description="A short snake_case label for the kind of change, e.g. "
                    "threshold_change, dollar_amount_change, deadline_change, "
                    "citation_update, definition_added, definition_removed, "
                    "scope_change, requirement_added, requirement_removed, "
                    "text_clarification, renumbering.",
    )
    description: str = Field(
        description="One concise sentence naming the concrete delta, including "
                    "the specific values, dates, thresholds, or citations "
                    "involved (e.g. 'Raised the micro-purchase threshold from "
                    "$10,000 to $15,000').",
    )


class SectionChanges(BaseModel):
    """The full set of logical changes for one DFARS section-unit."""
    changes: list[Change] = Field(
        default_factory=list,
        description="Every substantive logical change from BEFORE to AFTER. "
                    "Empty if nothing substantive changed.",
    )


EXTRACT_SYSTEM = """You are an expert analyst of U.S. federal acquisition \
regulations. You compare the before and after text of a DFARS section and \
enumerate the substantive changes that were made."""

EXTRACT_PROMPT = """\
Below is the official text of one DFARS section-unit BEFORE and AFTER an
amendment implementing an NDAA provision.

Identify the substantive *logical* changes from BEFORE to AFTER. Work out the
delta: text that is identical in both is unchanged boilerplate and must NOT be
reported. Group related edits into one logical change rather than reporting every
word tweak separately, but keep each change to a single, specific idea.

For each change, give:
- a short snake_case `category` (e.g. threshold_change, dollar_amount_change,
  deadline_change, citation_update, definition_added, definition_removed,
  scope_change, requirement_added, requirement_removed, text_clarification,
  renumbering); and
- a one-sentence `description` naming the concrete delta, including the specific
  values, dates, thresholds, citations, or defined terms involved.

Be concise and precise. Report only substantive changes (new/removed/altered
obligations, values, dates, citations, definitions, scope, or structure); ignore
pure formatting. If nothing substantive changed, return an empty list.

---

DFARS BEFORE:
\"\"\"
{before}
\"\"\"

DFARS AFTER:
\"\"\"
{after}
\"\"\"
"""


def extract_changes(before: str, after: str) -> list[dict]:
    """Extract the list of logical changes for one section-unit."""
    llm = _get_llm(temperature=0.0).with_structured_output(SectionChanges)
    result: SectionChanges = llm.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=EXTRACT_PROMPT.format(
            before=before if before.strip() else "(not available)",
            after=after,
        )),
    ])
    return [c.model_dump() for c in result.changes]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_inventory(limit: int = None, only_ndaa: str = None) -> dict:
    """Build the ``{<year>_<section>: {unit: {"changes": [...]}}}`` inventory."""
    expected = expected_section_units(single_ndaa=False, require_manifest=False)

    groups = sorted(expected.items())
    if only_ndaa is not None:
        groups = [g for g in groups if f"{g[0][0]}_{g[0][1]}" == only_ndaa]
        if not groups:
            raise SystemExit(f"No NDAA section matched --ndaa {only_ndaa!r}")
    if limit is not None:
        groups = groups[:limit]

    total_units = sum(len(secs) for _, secs in groups)
    print(f"Extracting changes for {len(groups)} NDAA section(s), "
          f"{total_units} section-unit(s) ...")

    inventory: dict = {}
    idx = 0
    for (year, section), secs in groups:
        key = f"{year}_{section}"
        unit_changes: dict = {}
        for unit, ba in sorted(secs.items()):
            idx += 1
            label = f"[{idx}/{total_units}] {year} s{section} -> {unit}"
            try:
                changes = extract_changes(ba.get("before", ""), ba.get("after", ""))
                print(f"  {label}  {len(changes)} change(s)")
            except Exception as exc:
                print(f"  {label}  ERROR: {exc}")
                changes = []
            unit_changes[unit] = {"changes": changes}
        inventory[key] = unit_changes

    return inventory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_FRAMEWORK_DIR / ".env")

    parser = argparse.ArgumentParser(
        description="Extract ground-truth logical changes per DFARS section-unit")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max NDAA sections to process (for test runs)")
    parser.add_argument("--ndaa", type=str, default=None,
                        help="Only process a single '<year>_<section>' group")
    args = parser.parse_args()

    inventory = build_inventory(limit=args.limit, only_ndaa=args.ndaa)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    n_units = sum(len(v) for v in inventory.values())
    n_changes = sum(len(u["changes"]) for v in inventory.values() for u in v.values())
    print(f"\nWrote {len(inventory)} NDAA section(s), {n_units} unit(s), "
          f"{n_changes} change(s) to {out_path}")


if __name__ == "__main__":
    main()
