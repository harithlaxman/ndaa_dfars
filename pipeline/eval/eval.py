"""
Assertion-based evaluation for the NDAA -> DFARS drafting pipelines.

Takes a run's results JSON and scores each NDAA group's draft against the
precomputed ground-truth change inventory (produced by ``prepare_eval.py``,
``data/eval/section_changes.json``). Instead of re-deriving the before->after
delta on every judge call, the judge is handed the expected *logical changes*
(assertions) for the group and only has to decide whether the draft satisfies
each one, classifying it into a scenario-anchored outcome:

  - captured_correct     present, right value, right place
  - captured_wrong_place present & right value, wrong subsection / wrong section
  - captured_wrong_value present, right place, wrong value/date/citation
  - missed               not found anywhere in the draft

Those outcomes map deterministically to the four scoring dimensions
(change_completeness, substantive_correctness, structural_fidelity, and
edit_minimality from unsupported changes). Plus a whole-draft BLEU.

The judge runs once per NDAA group over the whole draft, so it works for both
result formats:
  - baseline   -- a single ``draft``/``after`` blob per group
  - framework2 -- per-unit ``section_drafts`` (concatenated into a blob)

Usage (from the repo root)
--------------------------
  # Evaluate the latest pipeline results in data/results
  python pipeline/eval/eval.py

  # Use a specific results file
  python pipeline/eval/eval.py --input data/results/pipeline_baseline_results.json

  # Use a specific change inventory
  python pipeline/eval/eval.py --inventory data/eval/section_changes.json

  # Re-run the judge only (skip BLEU, keep existing scores)
  python pipeline/eval/eval.py --rejudge data/results/eval_results.json

  # Limit to N NDAA groups
  python pipeline/eval/eval.py --limit 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Literal

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sacrebleu.metrics import BLEU

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _EVAL_DIR.parents[0]        # pipeline/ -- for `agents.*` imports
_REPO_ROOT = _EVAL_DIR.parents[1]           # ndaa_dfars/ -- for data/
_FRAMEWORK_DIR = _PIPELINE_DIR / "agents" / "framework2"  # for .env
_DATA_DIR = _REPO_ROOT / "data"
sys.path.insert(0, str(_PIPELINE_DIR))

RESULTS_DIR = _DATA_DIR / "results"
DEFAULT_INVENTORY = _DATA_DIR / "eval" / "section_changes.json"


def latest_results_file() -> Path:
    """Most recent pipeline results JSON in data/results (baseline or 1:N)."""
    candidates = sorted(
        list(RESULTS_DIR.glob("pipeline_baseline_results*.json"))
        + list(RESULTS_DIR.glob("pipeline_1n_results*.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No pipeline_*_results*.json found in {RESULTS_DIR}; "
            "pass --input explicitly")
    return candidates[-1]


def load_inventory(path: str) -> dict:
    """Load the change inventory keyed `<year>_<section>` -> unit -> {changes}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Change inventory not found at {p}. Generate it first:\n"
            "  python pipeline/eval/prepare_eval.py")
    with open(p) as f:
        return json.load(f)


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
# Result-format adapter
# ---------------------------------------------------------------------------

def group_key(r: dict) -> str:
    return f'{r.get("ndaa_year")}_{r.get("ndaa_section")}'


def _labeled_blob(items: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"=== {sec} ===\n{text}" for sec, text in items if (text or "").strip()
    )


def group_view(r: dict) -> tuple[str, str]:
    """Reduce a result entry to its whole-group ``(draft_blob, after_blob)``.

    Handles both the baseline blob format and the framework2 ``section_drafts``
    format so a single judge code path scores either.
    """
    if "section_drafts" in r:  # framework2
        drafts = r.get("section_drafts", [])
        draft_blob = _labeled_blob(
            [(d.get("section", "?"), d.get("draft_clean", "")) for d in drafts])
        after_blob = _labeled_blob(
            [(d.get("section", "?"), d.get("after", "")) for d in drafts])
        return draft_blob, after_blob
    # baseline
    return r.get("draft", "") or "", r.get("after", "") or ""


# ---------------------------------------------------------------------------
# Assertion judge (one call per NDAA group)
# ---------------------------------------------------------------------------

Outcome = Literal[
    "captured_correct",
    "captured_wrong_place",
    "captured_wrong_value",
    "missed",
]


class AssertionVerdict(BaseModel):
    """The judge's verdict on a single expected change."""
    id: str = Field(
        description="The id of the assertion being judged (e.g. 'A3'), copied "
                    "exactly from the EXPECTED CHANGES list.",
    )
    outcome: Outcome = Field(
        description="captured_correct = present, right value, right place; "
                    "captured_wrong_place = present and right value but in the "
                    "wrong subsection or wrong DFARS section; "
                    "captured_wrong_value = present and in the right place but a "
                    "wrong value/date/citation/term; "
                    "missed = not found anywhere in the draft.",
    )
    found_in: Optional[str] = Field(
        default=None,
        description="If the change appears in a different section than its "
                    "target unit, the section it was actually found in; else null.",
    )
    reasoning: str = Field(
        description="One sentence citing the draft text that satisfies or "
                    "misses this change.",
    )


class GroupAssertionJudgement(BaseModel):
    """All verdicts for one NDAA group's draft, plus unsupported edits."""
    verdicts: list[AssertionVerdict] = Field(
        description="Exactly one verdict per expected change, identified by its "
                    "id. Include every id from the EXPECTED CHANGES list.",
    )
    unsupported_changes: list[str] = Field(
        default_factory=list,
        description="Substantive edits visible in the draft that do NOT "
                    "correspond to any listed expected change (one sentence each).",
    )
    reasoning: str = Field(
        default="",
        description="One short paragraph summarizing the draft's coverage.",
    )


JUDGE_SYSTEM = """You are an expert evaluator for regulatory text drafting \
systems. You are given the expected changes (assertions) that an official DFARS \
revision made, and a system's proposed draft. You decide, for each expected \
change, whether the draft made it."""

GROUP_ASSERTION_PROMPT = """\
You are scoring how well a system's proposed DFARS draft for one NDAA provision
captured the changes the official revision actually made. You are NOT given the
before/after text -- you are given the list of EXPECTED CHANGES (assertions),
already worked out, grouped by the DFARS section-unit each belongs to.

For EACH expected change, find it in the draft and classify it into exactly one
outcome:

- **captured_correct** — the change is present, the specific values/dates/
  citations/terms are right, and it is in the right section and numbering.
- **captured_wrong_place** — the change is present and its values are right, but
  it landed in the wrong subsection or in a different DFARS section than its
  target unit. Set `found_in` to the section it actually appears in.
- **captured_wrong_value** — the change is present and in the right place, but a
  value, date, threshold, citation, or defined term is wrong.
- **missed** — the change does not appear anywhere in the draft.

Return one verdict per expected change, copying its `id` exactly. Include every
id. Then list, in `unsupported_changes`, any substantive edits you can see in the
draft that do NOT correspond to any expected change (invented or unsupported
changes). Ignore pure formatting.

Judge the whole draft below; a change may appear in any section of it.

---

EXPECTED CHANGES (assertions), grouped by target unit:
{assertions_block}

---

SYSTEM DRAFT (all sections):
\"\"\"
{draft}
\"\"\"
"""


def flatten_assertions(assertions_by_unit: dict[str, list[dict]]) -> list[dict]:
    """Assign each expected change a stable id, flattened across units.

    Ids (A1, A2, ...) are the join key between the prompt, the model's verdicts,
    and the inventory (unit + category), so the model paraphrasing a description
    can't break the mapping.
    """
    flat = []
    i = 0
    for unit in sorted(assertions_by_unit):
        for c in assertions_by_unit[unit]:
            i += 1
            flat.append({
                "id": f"A{i}",
                "unit": unit,
                "category": c.get("category", ""),
                "description": c.get("description", ""),
            })
    return flat


def _assertions_block(flat: list[dict]) -> str:
    lines = []
    current_unit = None
    for a in flat:
        if a["unit"] != current_unit:
            current_unit = a["unit"]
            lines.append(f"Unit {current_unit}:")
        lines.append(f"  {a['id']}. [{a['category']}] {a['description']}")
    return "\n".join(lines)


def judge_group_assertions(
    draft_blob: str, flat: list[dict]
) -> GroupAssertionJudgement:
    """Run the assertion judge once over a whole group's draft."""
    llm = _get_llm(temperature=0.0).with_structured_output(GroupAssertionJudgement)
    prompt = GROUP_ASSERTION_PROMPT.format(
        assertions_block=_assertions_block(flat),
        draft=draft_blob if draft_blob.strip() else "(empty draft)",
    )
    return llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=prompt),
    ])


# ---------------------------------------------------------------------------
# Outcome -> score mapping (deterministic)
# ---------------------------------------------------------------------------

SECTION_SCORE_KEYS = [
    "change_completeness",
    "edit_minimality",
    "substantive_correctness",
    "structural_fidelity",
]

# Outcomes whose value (substance) is correct, and whose placement is correct.
_RIGHT_VALUE = {"captured_correct", "captured_wrong_place"}
_RIGHT_PLACE = {"captured_correct", "captured_wrong_value"}


def _frac_to_score(frac: float) -> int:
    """Map a 0..1 success fraction onto the 1-5 anchored scale."""
    return max(1, min(5, round(1 + 4 * frac)))


def _mean_scores(scores: dict) -> Optional[float]:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def _mean(vals: list) -> Optional[float]:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def score_from_verdicts(
    flat: list[dict],
    judgement: GroupAssertionJudgement,
) -> dict:
    """Turn the judge's verdicts into per-unit and per-group dimension scores.

    `flat` is the id-tagged assertion list (from `flatten_assertions`). Verdicts
    are joined back by id, so each assertion's authoritative unit/category come
    from the inventory, and any assertion the model omitted counts as `missed`.
    """
    unsupported = list(judgement.unsupported_changes)
    by_id = {a["id"]: a for a in flat}
    verdict_by_id = {v.id: v for v in judgement.verdicts if v.id in by_id}

    verdicts: list[dict] = []
    for a in flat:
        v = verdict_by_id.get(a["id"])
        verdicts.append({
            "id": a["id"],
            "unit": a["unit"],
            "category": a["category"],
            "description": a["description"],
            "outcome": v.outcome if v else "missed",
            "found_in": v.found_in if v else None,
            "reasoning": v.reasoning if v else "no verdict returned for this id",
        })

    per_unit: dict[str, dict] = {}
    units = {a["unit"] for a in flat}
    for unit in sorted(units):
        uv = [v for v in verdicts if v["unit"] == unit]
        total = len(uv)
        if total == 0:
            continue
        captured = [v for v in uv if v["outcome"] != "missed"]
        n_cap = len(captured)
        right_value = sum(1 for v in captured if v["outcome"] in _RIGHT_VALUE)
        right_place = sum(1 for v in captured if v["outcome"] in _RIGHT_PLACE)

        u = {
            "change_completeness": _frac_to_score(n_cap / total),
            "substantive_correctness": (
                _frac_to_score(right_value / n_cap) if n_cap else None),
            "structural_fidelity": (
                _frac_to_score(right_place / n_cap) if n_cap else None),
            "n_assertions": total,
            "n_captured": n_cap,
        }
        u["overall"] = _mean_scores({
            k: u[k] for k in
            ("change_completeness", "substantive_correctness", "structural_fidelity")
        })
        per_unit[unit] = u

    # edit_minimality is a whole-draft property: penalize unsupported changes.
    edit_minimality = max(1, 5 - 2 * len(unsupported))

    scores = {
        "change_completeness": _mean(
            [u["change_completeness"] for u in per_unit.values()]),
        "edit_minimality": edit_minimality,
        "substantive_correctness": _mean(
            [u["substantive_correctness"] for u in per_unit.values()]),
        "structural_fidelity": _mean(
            [u["structural_fidelity"] for u in per_unit.values()]),
    }
    scores["overall"] = _mean_scores(scores)

    return {
        "scores": scores,
        "per_unit": per_unit,
        "verdicts": verdicts,
        "unsupported_changes": unsupported,
        "reasoning": judgement.reasoning,
    }


def _floor_eval(flat: list[dict], note: str) -> dict:
    """All-missed eval for a group whose draft never reached the judge.

    Floors every dimension to 1 (a no-show must not dodge the average or score
    well on edit_minimality), and records a missed verdict per expected change.
    """
    verdicts = [
        {"id": a["id"], "unit": a["unit"], "category": a["category"],
         "description": a["description"], "outcome": "missed",
         "found_in": None, "reasoning": note}
        for a in flat
    ]
    scores = {k: 1 for k in SECTION_SCORE_KEYS}
    scores["overall"] = 1.0
    return {
        "scores": scores,
        "per_unit": {},
        "verdicts": verdicts,
        "unsupported_changes": [],
        "reasoning": note,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _has_draft(r: dict) -> bool:
    draft_blob, _ = group_view(r)
    return bool(draft_blob.strip())


def evaluate_results(
    results: list[dict], inventory: dict, skip_bleu: bool = False
) -> list[dict]:
    """Score each NDAA group's draft against the change inventory."""
    evaluated = []
    total = len(results)

    for i, r in enumerate(results, 1):
        key = group_key(r)
        assertions_by_unit = {
            unit: blk.get("changes", [])
            for unit, blk in inventory.get(key, {}).items()
        }
        flat = flatten_assertions(assertions_by_unit)
        label = f"[{i}/{total}] {r.get('ndaa_year')} s{r.get('ndaa_section')}"

        # Errored / undrafted group -> floor against the expected assertions.
        if "error" in r or not _has_draft(r):
            if flat:
                print(f"  {label}  (no draft) -> floored")
                evaluated.append({
                    **r, "assertion_eval": _floor_eval(
                        flat, "group not drafted (errored/empty)")})
            else:
                evaluated.append({
                    **r, "assertion_eval": {"skipped": "no draft, no assertions"}})
            continue

        if not flat:
            print(f"  {label}  no assertions in inventory, skipping")
            evaluated.append({
                **r, "assertion_eval": {"skipped": "no assertions in inventory"}})
            continue

        draft_blob, after_blob = group_view(r)

        # BLEU (whole draft vs whole ground truth)
        if skip_bleu:
            bleu = r.get("bleu")
            if bleu is None:
                bleu = compute_bleu(draft_blob, after_blob)
        else:
            bleu = compute_bleu(draft_blob, after_blob)

        try:
            judgement = judge_group_assertions(draft_blob, flat)
            assertion_eval = score_from_verdicts(flat, judgement)
            overall = assertion_eval["scores"].get("overall")
            n = len(assertion_eval["verdicts"])
            print(f"  {label}  BLEU={bleu:.1f}  Judge={overall}  ({n} assertions)")
        except Exception as exc:
            print(f"  {label}  Judge error: {exc}")
            assertion_eval = {"scores": {}, "reasoning": f"Judge error: {exc}",
                              "verdicts": [], "unsupported_changes": []}

        evaluated.append({**r, "bleu": bleu, "assertion_eval": assertion_eval})

    return evaluated


def rejudge_results(results: list[dict], inventory: dict) -> list[dict]:
    """Re-run the judge on previously evaluated results, keeping BLEU."""
    return evaluate_results(results, inventory, skip_bleu=True)


def penalize_missing(results: list[dict], inventory: dict) -> list[dict]:
    """Append floored entries for inventory groups entirely absent from results.

    Groups present but errored/undrafted are already floored in
    `evaluate_results`; this only covers groups the run never produced at all,
    so recall failures count instead of vanishing. Deterministic, no LLM calls.
    """
    present = {group_key(r) for r in results if "ndaa_year" in r}
    n_added = 0
    for key, units in inventory.items():
        if key in present:
            continue
        assertions_by_unit = {u: blk.get("changes", []) for u, blk in units.items()}
        flat = flatten_assertions(assertions_by_unit)
        if not flat:
            continue
        year, section = key.split("_", 1)
        results.append({
            "ndaa_year": year,
            "ndaa_section": section,
            "missing": True,
            "assertion_eval": _floor_eval(
                flat, "group not produced by the run (penalized)"),
        })
        n_added += 1
    if n_added:
        print(f"Penalized {n_added} NDAA group(s) absent from the run")
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print aggregate evaluation metrics."""
    evals = [
        r["assertion_eval"]
        for r in results
        if isinstance(r.get("assertion_eval"), dict)
        and "skipped" not in r["assertion_eval"]
        and r["assertion_eval"].get("scores")
    ]
    if not evals:
        print("\nNo evaluated groups found.")
        return

    print(f"\n{'='*72}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*72}")
    print(f"  NDAA groups evaluated: {len(evals)}")

    # BLEU
    bleu_scores = [r["bleu"] for r in results if isinstance(r.get("bleu"), (int, float))]
    if bleu_scores:
        avg = sum(bleu_scores) / len(bleu_scores)
        print(f"\n  {'BLEU':28s}  avg={avg:.1f}  "
              f"min={min(bleu_scores):.1f}  max={max(bleu_scores):.1f}  "
              f"n={len(bleu_scores)}")

    # Group dimension scores
    for key in SECTION_SCORE_KEYS + ["overall"]:
        values = [
            e["scores"][key] for e in evals
            if isinstance(e.get("scores", {}).get(key), (int, float))
        ]
        if values:
            avg = sum(values) / len(values)
            print(f"  {key:28s}  avg={avg:.2f}  "
                  f"min={min(values):.1f}  max={max(values):.1f}  "
                  f"n={len(values)}")

    # Outcome breakdown
    outcomes: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    for e in evals:
        for v in e.get("verdicts", []):
            outcomes[v["outcome"]] = outcomes.get(v["outcome"], 0) + 1
            cat = v.get("category") or "uncategorized"
            d = by_category.setdefault(cat, {})
            d[v["outcome"]] = d.get(v["outcome"], 0) + 1
    if outcomes:
        total = sum(outcomes.values())
        print(f"\n  Assertion outcomes (n={total}):")
        for o in ("captured_correct", "captured_wrong_value",
                  "captured_wrong_place", "missed"):
            if o in outcomes:
                print(f"    {o:22s} {outcomes[o]:4d}  "
                      f"({100*outcomes[o]/total:.0f}%)")

    if by_category:
        print(f"\n  Capture rate by change category:")
        for cat in sorted(by_category):
            d = by_category[cat]
            tot = sum(d.values())
            captured = tot - d.get("missed", 0)
            print(f"    {cat:24s} {captured:3d}/{tot:<3d} captured "
                  f"({100*captured/tot:.0f}%)")

    # Per-group overview
    print(f"\n  {'NDAA':<16} {'BLEU':>6} {'Cmpl':>5} {'Subs':>5} "
          f"{'Strc':>5} {'Min':>5} {'Ovr':>5}")
    print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for r in results:
        e = r.get("assertion_eval")
        if not isinstance(e, dict) or not e.get("scores"):
            continue
        s = e["scores"]
        ndaa = f"{r.get('ndaa_year')} s{r.get('ndaa_section')}"
        bleu = r.get("bleu")
        bleu_s = f"{bleu:6.1f}" if isinstance(bleu, (int, float)) else f"{'-':>6}"

        def fmt(x):
            return f"{x:5.1f}" if isinstance(x, (int, float)) else f"{'-':>5}"
        print(f"  {ndaa:<16} {bleu_s} {fmt(s.get('change_completeness'))} "
              f"{fmt(s.get('substantive_correctness'))} "
              f"{fmt(s.get('structural_fidelity'))} "
              f"{fmt(s.get('edit_minimality'))} {fmt(s.get('overall'))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_FRAMEWORK_DIR / ".env")

    parser = argparse.ArgumentParser(
        description="Assertion-based evaluation of NDAA->DFARS pipeline results")
    parser.add_argument("--input", type=str, default=None,
                        help="Pipeline results JSON (default: latest "
                             "pipeline_*_results*.json in data/results)")
    parser.add_argument("--inventory", type=str, default=str(DEFAULT_INVENTORY),
                        help=f"Change inventory JSON (default: {DEFAULT_INVENTORY})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/results/"
                             "eval_results_<timestamp>.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max NDAA groups to evaluate")
    parser.add_argument("--rejudge", type=str, default=None,
                        help="Path to previous eval results — re-run judge only")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(RESULTS_DIR / f"eval_results_{ts}.json")

    inventory = load_inventory(args.inventory)
    print(f"Loaded inventory: {len(inventory)} NDAA group(s) from {args.inventory}")

    if args.rejudge:
        print(f"Re-judging from {args.rejudge} ...")
        with open(args.rejudge) as f:
            results = json.load(f)
        if args.limit:
            results = results[:args.limit]
        results = rejudge_results(results, inventory)
    else:
        input_path = args.input or str(latest_results_file())
        print(f"Loading results from {input_path} ...")
        with open(input_path) as f:
            results = json.load(f)
        if args.limit:
            results = results[:args.limit]
        print(f"  {len(results)} NDAA groups, evaluating ...")
        results = evaluate_results(results, inventory)

    # Penalize inventory groups the run never produced at all.
    results = penalize_missing(results, inventory)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
