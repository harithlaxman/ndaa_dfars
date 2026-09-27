"""
Split evaluation scores by how many DFARS sections an NDAA provision changed.

For each NDAA group we count the "size" N = the number of DFARS sections whose
ground-truth text actually changed (before != after at the word level; sections
the NDAA left unchanged do not count). We then plot two views against N:

  1. Section-judge means — per-section BLEU and LLM-judge dimensions, POOLED
     over every changed section belonging to groups of size N. Answers: does
     per-section draft quality degrade as a provision fans out to more sections?

  2. Group coordination — the group_eval cross-section dimensions, averaged
     over the GROUPS of size N (one coordination score per group). Answers:
     does cross-section coordination get harder as N grows? This is the metric
     the 1:N architecture is meant to move.

Usage (from the repo root)
--------------------------
  # Latest eval results in data/results -> writes two PNGs next to it
  python pipeline/agents/framework2/score_by_group_size.py

  # A specific eval results file
  python pipeline/agents/framework2/score_by_group_size.py \
      --input data/results/eval_1n_results.json

  # Custom output prefix / format, and lump groups with N >= 6 into "6+"
  python pipeline/agents/framework2/score_by_group_size.py \
      --out-prefix figs/by_group_size --ext pdf --max-bin 6
"""

import argparse
import json
import re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = _DATA_DIR / "results"

# Reuse the section-judge keys and helpers from the change-type scorer.
from score_by_change_type import JUDGE_KEYS, _avg, latest_eval_file  # noqa: E402

GROUP_KEYS = [
    "definition_placement",
    "cross_section_consistency",
    "cross_reference_validity",
    "delegation_correctness",
    "overall",
]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _section_changed(draft: dict) -> bool:
    """True if this section's ground truth actually changed."""
    before, after = draft.get("before"), draft.get("after")
    if not after:
        return False
    return _normalize_ws(before) != _normalize_ws(after)


def collect_groups(results: list[dict]) -> list[dict]:
    """One row per NDAA group: its changed-section count N, the pooled
    section-judge scores for those changed sections, and the group_eval
    coordination scores."""
    groups = []
    for r in results:
        if "error" in r:
            continue
        changed = [d for d in r.get("section_drafts", [])
                   if _section_changed(d)]
        n = len(changed)
        if n == 0:
            continue

        section_scores = []
        for d in changed:
            ev = d.get("eval", {})
            if not ev or "skipped" in ev:
                continue
            scores = ev.get("judge", {}).get("scores", {})
            section_scores.append({
                "bleu": ev.get("bleu"),
                **{k: scores.get(k) for k in JUDGE_KEYS},
            })

        ge = r.get("group_eval")
        group_scores = (ge.get("scores", {})
                        if isinstance(ge, dict) and "skipped" not in ge else {})

        groups.append({
            "ndaa": f"{r['ndaa_year']} s{r['ndaa_section']}",
            "n_changed": n,
            "section_scores": section_scores,
            "group_scores": group_scores,
        })
    return groups


def _binned(value: int, max_bin: int | None) -> str:
    """Label for an N value, lumping N >= max_bin into 'max_bin+'."""
    if max_bin is not None and value >= max_bin:
        return f"{max_bin}+"
    return str(value)


def _bin_order(labels: set[str]) -> list[str]:
    """Sort bin labels numerically, keeping a trailing '+' bucket last."""
    return sorted(labels, key=lambda s: (int(s.rstrip("+")), s.endswith("+")))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_lines(ax, x_labels, series, ylabel, ylim, counts):
    """Draw one metric per line over the binned x-axis."""
    for name, ys in series.items():
        xs = [i for i, y in enumerate(ys) if y is not None]
        vals = [y for y in ys if y is not None]
        ax.plot(xs, vals, marker="o", label=name)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(
        [f"{lbl}\n(g={c})" for lbl, c in zip(x_labels, counts)])
    ax.set_xlabel("sections changed per NDAA provision (N)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)


def plot_section_means(groups: list[dict], out_path: Path,
                       max_bin: int | None = None) -> Path:
    """Section-judge means (judges on 1-5, BLEU on a twin 0-100 axis) vs N,
    pooling all changed sections from groups of each size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled: dict[str, list[dict]] = {}
    for g in groups:
        pooled.setdefault(_binned(g["n_changed"], max_bin), []).extend(
            g["section_scores"])
    pooled = {k: v for k, v in pooled.items() if v}
    if not pooled:
        raise ValueError("No changed sections with scores to plot.")

    x_labels = _bin_order(set(pooled))
    counts = [len(pooled[lbl]) for lbl in x_labels]  # sections pooled per bin

    judge_series = {
        k: [_avg([s.get(k) for s in pooled[lbl]]) for lbl in x_labels]
        for k in JUDGE_KEYS
    }
    bleu_series = [_avg([s.get("bleu") for s in pooled[lbl]])
                   for lbl in x_labels]

    fig, ax = plt.subplots(figsize=(1.3 * len(x_labels) + 4, 5))
    _plot_lines(ax, x_labels, judge_series,
                ylabel="judge score (1-5)", ylim=(1, 5), counts=counts)
    ax.set_title("Section-judge scores vs. sections changed\n"
                 "(pooled over changed sections; g = # sections in bin)")

    ax2 = ax.twinx()
    xs = [i for i, y in enumerate(bleu_series) if y is not None]
    ax2.plot(xs, [y for y in bleu_series if y is not None],
             marker="s", linestyle="--", color="gray", label="BLEU")
    ax2.set_ylabel("BLEU (0-100)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_coordination(groups: list[dict], out_path: Path,
                            max_bin: int | None = None) -> Path:
    """Group_eval coordination dimensions vs N, averaged over groups of each
    size. Only groups that received a coordination score (N >= 2) appear."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binned: dict[str, list[dict]] = {}
    for g in groups:
        if g["group_scores"]:
            binned.setdefault(_binned(g["n_changed"], max_bin), []).append(
                g["group_scores"])
    binned = {k: v for k, v in binned.items() if v}
    if not binned:
        raise ValueError("No groups with coordination scores to plot.")

    x_labels = _bin_order(set(binned))
    counts = [len(binned[lbl]) for lbl in x_labels]  # groups per bin

    series = {
        k: [_avg([gs.get(k) for gs in binned[lbl]]) for lbl in x_labels]
        for k in GROUP_KEYS
    }

    fig, ax = plt.subplots(figsize=(1.3 * len(x_labels) + 4, 5))
    _plot_lines(ax, x_labels, series,
                ylabel="coordination score (1-5)", ylim=(1, 5), counts=counts)
    ax.set_title("Group coordination scores vs. sections changed\n"
                 "(averaged over groups; g = # groups in bin)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_split(groups: list[dict], max_bin: int | None = None) -> None:
    pooled: dict[str, list[dict]] = {}
    coord: dict[str, list[dict]] = {}
    for g in groups:
        lbl = _binned(g["n_changed"], max_bin)
        pooled.setdefault(lbl, []).extend(g["section_scores"])
        if g["group_scores"]:
            coord.setdefault(lbl, []).append(g["group_scores"])

    metrics = ["bleu"] + JUDGE_KEYS
    header = (f"  {'N changed':<10} {'groups':>6} {'secs':>5}"
              + "".join(f" {m[:12]:>12}" for m in metrics))
    print(header)
    print("  " + "-" * (len(header) - 2))
    n_groups = {lbl: 0 for lbl in pooled}
    for g in groups:
        n_groups[_binned(g["n_changed"], max_bin)] += 1
    for lbl in _bin_order(set(pooled)):
        secs = pooled[lbl]
        cells = "".join(
            f" {(_avg([s.get(m) for s in secs]) or float('nan')):>12.{1 if m=='bleu' else 2}f}"
            for m in metrics)
        print(f"  {lbl:<10} {n_groups[lbl]:>6} {len(secs):>5}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot eval scores by number of DFARS sections changed")
    parser.add_argument("--input", type=str, default=None,
                        help="Eval results JSON (default: latest "
                             "eval_1n_results*.json in data/results)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output path prefix (default: <input>_by_group_size)")
    parser.add_argument("--ext", type=str, default="png",
                        help="Image format/extension (png, pdf, svg). Default png")
    parser.add_argument("--max-bin", type=int, default=None,
                        help="Lump groups with N >= MAX_BIN into one 'MAX_BIN+' bin")
    args = parser.parse_args()

    input_path = args.input or str(latest_eval_file())
    print(f"Loading eval results from {input_path} ...")
    with open(input_path) as f:
        results = json.load(f)

    groups = collect_groups(results)
    if not groups:
        print("No NDAA groups with changed sections found.")
        return
    print(f"  {len(groups)} NDAA groups with >=1 changed section\n")
    print_split(groups, max_bin=args.max_bin)

    prefix = args.out_prefix or str(
        Path(input_path).with_name(Path(input_path).stem + "_by_group_size"))
    ext = args.ext.lstrip(".")
    p1 = plot_section_means(groups, f"{prefix}_sections.{ext}", args.max_bin)
    p2 = plot_group_coordination(groups, f"{prefix}_coordination.{ext}",
                                 args.max_bin)
    print(f"\n  Section-means plot saved to {p1}")
    print(f"  Coordination plot saved to {p2}")


if __name__ == "__main__":
    main()
