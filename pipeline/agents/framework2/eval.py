"""
Evaluation pipeline for the 1:N NDAA -> DFARS drafting framework.

Takes the results JSON produced by agent.py and evaluates each section draft
against the ground-truth "after" text using:
  1. BLEU score (sacrebleu, sentence-level)
  2. Section-level LLM-as-a-judge (4-dimension anchored rubric, 1-5 scale,
     delta-focused: the judge sees the "before" text so it scores the changes
     rather than the shared boilerplate)
  3. Group-level LLM-as-a-judge (one call per NDAA group, scoring the
     cross-section coordination that the 1:N architecture is supposed to buy:
     definition placement, consistency, cross-references, delegation)

Usage (from the repo root)
--------------------------
  # Evaluate the latest pipeline results in data/results
  python pipeline/agents/framework2/eval.py

  # Use a specific results file
  python pipeline/agents/framework2/eval.py --input data/results/pipeline_1n_results.json

  # Re-run only the judges (skip BLEU, keep existing scores)
  python pipeline/agents/framework2/eval.py --rejudge data/results/eval_1n_results.json

  # Limit to N NDAA groups
  python pipeline/agents/framework2/eval.py --limit 5

  # Only evaluate sections from single-NDAA document numbers
  python pipeline/agents/framework2/eval.py --single-ndaa
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sacrebleu.metrics import BLEU

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FRAMEWORK_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _FRAMEWORK_DIR.parents[1]   # pipeline/ -- for `agents.*` imports
_REPO_ROOT = _FRAMEWORK_DIR.parents[2]      # ndaa_dfars/ -- for data/
_DATA_DIR = _REPO_ROOT / "data"
sys.path.insert(0, str(_PIPELINE_DIR))

RESULTS_DIR = _DATA_DIR / "results"

from agents.framework2.utils import get_single_ndaa_allowed  # noqa: E402


def latest_results_file() -> Path:
    """Most recent pipeline results JSON in data/results."""
    candidates = sorted(RESULTS_DIR.glob("pipeline_1n_results*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No pipeline_1n_results*.json found in {RESULTS_DIR}; "
            "pass --input explicitly")
    return candidates[-1]


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
# BLEU
# ---------------------------------------------------------------------------

_bleu = BLEU(effective_order=True)


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU between hypothesis and reference."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    result = _bleu.sentence_score(hypothesis.strip(), [reference.strip()])
    return result.score


# ---------------------------------------------------------------------------
# Judge schemas (structured output)
# ---------------------------------------------------------------------------

class SectionJudgement(BaseModel):
    """Per-section judge scores, 1-5 scale."""
    change_completeness: int = Field(
        ge=1, le=5,
        description="Did the draft make every change the ground truth made?",
    )
    edit_minimality: int = Field(
        ge=1, le=5,
        description="Did the draft avoid changes the ground truth did not make?",
    )
    substantive_correctness: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="For changes the draft attempted: are values, dates, "
                    "thresholds, citations, and defined terms exactly right? "
                    "null if the draft attempted no changes.",
    )
    structural_fidelity: int = Field(
        ge=1, le=5,
        description="Numbering, paragraph hierarchy, definitions placement "
                    "consistent with the ground truth.",
    )
    reasoning: str = Field(
        description="One paragraph explaining the scores, citing the specific "
                    "changes that were captured, missed, or invented.",
    )


class GroupJudgement(BaseModel):
    """Per-NDAA-group judge scores over all section drafts, 1-5 scale.
    Score null for any dimension that does not apply to this group."""
    definition_placement: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="Definitions hosted where the ground truth hosts them, "
                    "once, with cross-references elsewhere — not duplicated "
                    "across sections. null if no definitions are involved.",
    )
    cross_section_consistency: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="Terms, thresholds, deadlines, and effective dates agree "
                    "across all drafts; no contradictions.",
    )
    cross_reference_validity: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="Every cross-reference between drafted sections points at "
                    "a section that actually contains the referenced material. "
                    "null if no cross-references are involved.",
    )
    delegation_correctness: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="Substantive changes landed in the sections where the "
                    "ground truth made them; sections the ground truth left "
                    "alone were left alone.",
    )
    reasoning: str = Field(
        description="One paragraph explaining the scores, citing specific "
                    "sections.",
    )


SECTION_SCORE_KEYS = [
    "change_completeness",
    "edit_minimality",
    "substantive_correctness",
    "structural_fidelity",
]

GROUP_SCORE_KEYS = [
    "definition_placement",
    "cross_section_consistency",
    "cross_reference_validity",
    "delegation_correctness",
]


def _mean_scores(scores: dict) -> Optional[float]:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


# ---------------------------------------------------------------------------
# Section-level LLM-as-a-Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert evaluator for regulatory text drafting systems.
You compare a system's proposed DFARS revision against the official ground-truth
revision and score how well the system captured the required changes."""

SECTION_JUDGE_PROMPT = """\
You are evaluating a system that reads an NDAA provision and the current DFARS
text, then proposes a revised DFARS section.

Below you have:
1. The DFARS text BEFORE the change. Both the system and the official drafters
   started from this exact text.
2. The official DFARS text AFTER the change (ground truth).
3. The system's proposed draft (clean, with changes applied).

Score the DELTA, not the document: first work out what the ground truth changed
(BEFORE vs AFTER), then check whether the draft made those same changes
(BEFORE vs DRAFT). Text that is identical in all three is shared boilerplate
and is not evidence of quality.

---

DFARS BEFORE (common starting text):
\"\"\"
{before}
\"\"\"

DFARS AFTER (ground truth):
\"\"\"
{after}
\"\"\"

SYSTEM DRAFT (proposed revision):
\"\"\"
{draft}
\"\"\"

---

Score each dimension on a 1-5 scale:

1. **change_completeness** — Did the draft make every change the ground truth made?
   5 = every ground-truth change is present in the draft
   4 = all substantive changes present; a minor edit (wording, renumbering) missed
   3 = the main change is present but a secondary change is missing
   2 = most ground-truth changes are missing, though some attempt was made
   1 = none of the ground-truth changes were made (draft is unchanged or unrelated)

2. **edit_minimality** — Did the draft avoid changes the ground truth did NOT make?
   5 = no changes beyond the ground truth's
   4 = trivial extra edits only (formatting, equivalent rephrasing)
   3 = one substantive extra change not present in the ground truth
   2 = several unsupported changes or noticeable rewriting of stable text
   1 = the draft largely invents content or rewrites text wholesale

3. **substantive_correctness** — For the changes the draft DID attempt: are the
   specifics exactly right — thresholds, dollar amounts, deadlines, dates,
   percentages, statutory/regulatory citations (USC, Public Law, CFR, DFARS),
   and defined terms?
   5 = all values and citations exact
   3 = one wrong value or citation in an otherwise correct change
   1 = the key values or citations are wrong
   null = the draft attempted no changes, so there is nothing to score

4. **structural_fidelity** — Does the draft preserve the regulatory structure
   (subsection numbering, paragraph organization, definitions placement)
   consistent with the ground truth?
   5 = structure matches the ground truth
   3 = right content placed under wrong numbering or in the wrong paragraph
   1 = structure substantially broken or reorganized

In `reasoning`, name the specific changes that were captured, missed, or
invented.
"""


def judge_draft(before: str, after: str, draft: str) -> dict:
    """Judge a single section draft against ground truth (delta-focused)."""
    llm = _get_llm(temperature=0.0).with_structured_output(SectionJudgement)

    prompt = SECTION_JUDGE_PROMPT.format(
        before=before if before.strip() else "(not available)",
        after=after,
        draft=draft,
    )
    result: SectionJudgement = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    scores = {k: getattr(result, k) for k in SECTION_SCORE_KEYS}
    scores["overall"] = _mean_scores(scores)
    return {"scores": scores, "reasoning": result.reasoning}


# ---------------------------------------------------------------------------
# Group-level LLM-as-a-Judge
# ---------------------------------------------------------------------------

GROUP_JUDGE_PROMPT = """\
You are evaluating how well a multi-section drafting system COORDINATED its
changes across all the DFARS sections affected by one NDAA provision. Each
section was drafted with an assigned role (primary / secondary / cite-only /
unaffected). The individual drafts have already been scored separately — here
you only score the cross-section properties.

For each section below you have the common BEFORE text, the official ground
truth AFTER, and the system's DRAFT.

{sections_block}

---

Score each dimension on a 1-5 scale, or null where the dimension does not
apply to this group:

1. **definition_placement** — Definitions appear once, in the section where the
   ground truth hosts them, with other sections cross-referencing rather than
   restating them. Duplicated or misplaced definitions lower the score.
   null if the changes involve no definitions.

2. **cross_section_consistency** — Terms, thresholds, deadlines, and effective
   dates agree across all drafts; no draft contradicts another.

3. **cross_reference_validity** — Every reference a draft makes to another
   drafted section points at a section that actually contains the referenced
   material (clause, definition, or prescription), matching how the ground
   truth wires the sections together. null if no cross-references are involved.

4. **delegation_correctness** — The substantive changes landed in the same
   sections where the ground truth made them; sections the ground truth left
   unchanged were left unchanged.

In `reasoning`, cite the specific sections behind each score.
"""

GROUP_SECTION_TEMPLATE = """\
=== SECTION {section} (assigned role: {role}) ===

BEFORE:
\"\"\"
{before}
\"\"\"

AFTER (ground truth):
\"\"\"
{after}
\"\"\"

DRAFT:
\"\"\"
{draft}
\"\"\"
"""


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def judge_group(drafts: list[dict]) -> dict:
    """Judge cross-section coordination for one NDAA group."""
    sections_block = "\n".join(
        GROUP_SECTION_TEMPLATE.format(
            section=d.get("section", "?"),
            role=d.get("role", "?"),
            before=d.get("before", "") or "(not available)",
            after=d.get("after", ""),
            draft=d.get("draft_clean", ""),
        )
        for d in drafts
    )

    llm = _get_llm(temperature=0.0).with_structured_output(GroupJudgement)
    result: GroupJudgement = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=GROUP_JUDGE_PROMPT.format(
            sections_block=sections_block)),
    ])

    scores = {k: getattr(result, k) for k in GROUP_SCORE_KEYS}
    scores["overall"] = _mean_scores(scores)

    # Deterministic delegation check: sections the agent marked "unaffected"
    # whose ground truth actually changed.
    unaffected_but_changed = [
        d["section"]
        for d in drafts
        if d.get("role") == "unaffected"
        and d.get("before") and d.get("after")
        and _normalize_ws(d["before"]) != _normalize_ws(d["after"])
    ]

    return {
        "scores": scores,
        "reasoning": result.reasoning,
        "unaffected_but_changed": unaffected_but_changed,
    }


# ---------------------------------------------------------------------------
# Backfill "before" text for results produced before agent.py attached it
# ---------------------------------------------------------------------------

def backfill_before(results: list[dict]) -> None:
    """Attach ground-truth 'before' text to drafts that lack it, using the
    same NDAA groups the pipeline was run from. Mutates results in place."""
    missing = any(
        not d.get("before")
        for r in results if "error" not in r
        for d in r.get("section_drafts", [])
    )
    if not missing:
        return

    print("Backfilling 'before' text from NDAA groups ...")
    from agents.framework2.agent import load_ndaa_groups
    before_map = {
        (str(g["ndaa"]["year"]), str(g["ndaa"]["section"]), s["section"]):
            s["before"]
        for g in load_ndaa_groups()
        for s in g["dfars_sections"]
    }

    unmatched = 0
    for r in results:
        if "error" in r:
            continue
        for d in r.get("section_drafts", []):
            if d.get("before"):
                continue
            key = (str(r["ndaa_year"]), str(r["ndaa_section"]), d.get("section"))
            d["before"] = before_map.get(key, "")
            if not d["before"]:
                unmatched += 1
    if unmatched:
        print(f"  WARNING: no 'before' text found for {unmatched} draft(s); "
              "their judges will run without the delta context")


# ---------------------------------------------------------------------------
# Single-NDAA filter
# ---------------------------------------------------------------------------

def filter_single_ndaa(results: list[dict]) -> list[dict]:
    """Keep only section drafts whose DFARS sections belong to document numbers
    with exactly one NDAA citation in doc_to_ndaa.csv."""
    allowed = get_single_ndaa_allowed()

    filtered = []
    for r in results:
        ndaa_key = (str(r["ndaa_year"]), str(r["ndaa_section"]))
        secs = allowed.get(ndaa_key)
        if not secs:
            continue

        kept = [
            d for d in r.get("section_drafts", [])
            if any(s in d.get("section", "") for s in secs)
        ]
        if kept:
            filtered.append({**r, "section_drafts": kept})

    return filtered


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _judge_group_for_result(r: dict, eval_drafts: list[dict]) -> Optional[dict]:
    """Run the group judge for one NDAA group; None if it doesn't apply."""
    judgeable = [d for d in eval_drafts if d.get("after")]
    if len(judgeable) < 2:
        return {"skipped": "fewer than 2 sections with ground truth"}
    try:
        group_eval = judge_group(judgeable)
        overall = group_eval["scores"].get("overall")
        print(f"  Group judge ({len(judgeable)} sections)  Overall={overall}")
        return group_eval
    except Exception as exc:
        print(f"  Group judge error: {exc}")
        return {"scores": {}, "reasoning": f"Group judge error: {exc}",
                "unaffected_but_changed": []}


def evaluate_results(results: list[dict], skip_bleu: bool = False) -> list[dict]:
    """Evaluate each section draft, then each NDAA group, in the results."""
    backfill_before(results)

    evaluated = []
    total_sections = sum(
        len(r.get("section_drafts", []))
        for r in results
        if "error" not in r
    )
    section_idx = 0

    for r in results:
        if "error" in r:
            evaluated.append(r)
            continue

        eval_drafts = []
        for d in r.get("section_drafts", []):
            section_idx += 1
            section = d.get("section", "?")
            label = (
                f"[{section_idx}/{total_sections}] "
                f"NDAA {r['ndaa_year']} s{r['ndaa_section']} -> {section}"
            )

            draft_clean = d.get("draft_clean", "")
            before = d.get("before", "")
            after = d.get("after", "")

            if not after:
                print(f"  {label} — no ground truth, skipping eval")
                eval_drafts.append({**d, "eval": {"skipped": "no ground truth"}})
                continue

            # BLEU
            if skip_bleu:
                bleu = d.get("eval", {}).get("bleu")
                if bleu is None:
                    bleu = compute_bleu(draft_clean, after)
            else:
                bleu = compute_bleu(draft_clean, after)
            print(f"  {label}  BLEU={bleu:.1f}", end="")

            # Section judge
            try:
                judgement = judge_draft(before, after, draft_clean)
                overall = judgement["scores"].get("overall", "?")
                print(f"  Judge={overall}")
            except Exception as exc:
                print(f"  Judge error: {exc}")
                judgement = {
                    "scores": {},
                    "reasoning": f"Judge error: {exc}",
                }

            eval_drafts.append({
                **d,
                "eval": {
                    "bleu": bleu,
                    "judge": judgement,
                },
            })

        evaluated.append({
            **r,
            "section_drafts": eval_drafts,
            "group_eval": _judge_group_for_result(r, eval_drafts),
        })

    return evaluated


def rejudge_results(results: list[dict]) -> list[dict]:
    """Re-run only the LLM judges on previously evaluated results,
    keeping existing BLEU scores."""
    return evaluate_results(results, skip_bleu=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print aggregate evaluation metrics."""
    all_evals = [
        d["eval"]
        for r in results if "error" not in r
        for d in r.get("section_drafts", [])
        if "eval" in d and "skipped" not in d["eval"]
    ]

    if not all_evals:
        print("\nNo evaluated sections found.")
        return

    print(f"\n{'='*72}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*72}")
    print(f"  Sections evaluated: {len(all_evals)}")

    # BLEU
    bleu_scores = [e["bleu"] for e in all_evals if "bleu" in e]
    if bleu_scores:
        avg = sum(bleu_scores) / len(bleu_scores)
        print(f"\n  {'BLEU':25s}  avg={avg:.1f}  "
              f"min={min(bleu_scores):.1f}  max={max(bleu_scores):.1f}  "
              f"n={len(bleu_scores)}")

    # Section judge scores
    for key in SECTION_SCORE_KEYS + ["overall"]:
        values = [
            e["judge"]["scores"][key]
            for e in all_evals
            if isinstance(
                e.get("judge", {}).get("scores", {}).get(key), (int, float))
        ]
        if values:
            avg = sum(values) / len(values)
            print(f"  {key:25s}  avg={avg:.2f}  "
                  f"min={min(values):.1f}  max={max(values):.1f}  "
                  f"n={len(values)}")

    # Group judge scores
    group_evals = [
        r["group_eval"]
        for r in results
        if "error" not in r
        and isinstance(r.get("group_eval"), dict)
        and "skipped" not in r["group_eval"]
    ]
    if group_evals:
        print(f"\n  NDAA groups judged: {len(group_evals)}")
        for key in GROUP_SCORE_KEYS + ["overall"]:
            values = [
                g["scores"][key]
                for g in group_evals
                if isinstance(g.get("scores", {}).get(key), (int, float))
            ]
            if values:
                avg = sum(values) / len(values)
                print(f"  {key:25s}  avg={avg:.2f}  "
                      f"min={min(values):.1f}  max={max(values):.1f}  "
                      f"n={len(values)}")
        flagged = [s for g in group_evals
                   for s in g.get("unaffected_but_changed", [])]
        if flagged:
            print(f"\n  Sections marked 'unaffected' whose ground truth "
                  f"changed ({len(flagged)}):")
            for s in flagged:
                print(f"    - {s}")

    # Per-section overview
    print(f"\n  {'NDAA':<20} {'Section':<35} {'BLEU':>6} {'Judge':>6}")
    print(f"  {'-'*20} {'-'*35} {'-'*6} {'-'*6}")
    for r in results:
        if "error" in r:
            continue
        for d in r.get("section_drafts", []):
            ev = d.get("eval", {})
            if "skipped" in ev:
                continue
            ndaa = f"{r['ndaa_year']} s{r['ndaa_section']}"
            sec = (d.get("section") or "")[:35]
            bleu = ev.get("bleu", 0)
            overall = ev.get("judge", {}).get("scores", {}).get("overall", "?")
            print(f"  {ndaa:<20} {sec:<35} {bleu:>6.1f} {overall:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_FRAMEWORK_DIR / ".env")

    parser = argparse.ArgumentParser(
        description="Evaluate 1:N NDAA->DFARS pipeline results")
    parser.add_argument("--input", type=str, default=None,
                        help="Pipeline results JSON (default: latest "
                             "pipeline_1n_results*.json in data/results)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/results/"
                             "eval_1n_results_<timestamp>.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max NDAA groups to evaluate")
    parser.add_argument("--rejudge", type=str, default=None,
                        help="Path to previous eval results — re-run judges only")
    parser.add_argument("--single-ndaa", action="store_true",
                        help="Only evaluate sections from document numbers "
                             "with exactly one NDAA citation")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(RESULTS_DIR / f"eval_1n_results_{ts}.json")

    if args.rejudge:
        print(f"Re-judging from {args.rejudge} ...")
        with open(args.rejudge) as f:
            prev = json.load(f)
        if args.single_ndaa:
            prev = filter_single_ndaa(prev)
            print(f"  Filtered to {len(prev)} NDAA groups (single-NDAA docs)")
        if args.limit:
            prev = prev[:args.limit]
        results = rejudge_results(prev)
    else:
        input_path = args.input or str(latest_results_file())
        print(f"Loading results from {input_path} ...")
        with open(input_path) as f:
            results = json.load(f)
        if args.single_ndaa:
            results = filter_single_ndaa(results)
            print(f"  Filtered to {len(results)} NDAA groups (single-NDAA docs)")
        if args.limit:
            results = results[:args.limit]
        print(f"  {len(results)} NDAA groups, evaluating ...")
        results = evaluate_results(results)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
