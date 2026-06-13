"""
Split evaluation scores by the type of ground-truth change to the DFARS section.

Classifies each evaluated section draft by diffing the ground-truth "before"
vs "after" text (word level):
  - addition      almost all changed tokens were added
  - deletion      almost all changed tokens were removed
  - modification  a mix of added and removed tokens
  - unchanged     before == after

then prints average BLEU and LLM-judge scores per type.

Usage (from the repo root)
--------------------------
  # Latest eval results in data/results
  python pipeline/agents/framework2/score_by_change_type.py

  # A specific eval results file
  python pipeline/agents/framework2/score_by_change_type.py --input data/results/eval_1n_results.json

  # Also list every section with its type
  python pipeline/agents/framework2/score_by_change_type.py --per-section
"""

import argparse
import difflib
import json
import re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = _DATA_DIR / "results"

JUDGE_KEYS = [
    "change_completeness",
    "edit_minimality",
    "substantive_correctness",
    "structural_fidelity",
    "overall",
]

CHANGE_TYPES = ["addition", "modification", "deletion", "unchanged"]

# A change counts as pure addition/deletion if the other side contributes
# under this fraction of the changed tokens.
PURE_THRESHOLD = 0.1


def latest_eval_file() -> Path:
    candidates = sorted(RESULTS_DIR.glob("eval_1n_results*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No eval_1n_results*.json found in {RESULTS_DIR}; "
            "pass --input explicitly")
    return candidates[-1]


def _tokens(text: str) -> list[str]:
    return re.sub(r"\s+", " ", text).strip().split(" ")


def classify_change(before: str, after: str) -> dict:
    """Classify the ground-truth change by word-level diff."""
    added = removed = 0
    matcher = difflib.SequenceMatcher(None, _tokens(before), _tokens(after),
                                      autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in ("insert", "replace"):
            added += j2 - j1
        if op in ("delete", "replace"):
            removed += i2 - i1

    total = added + removed
    if total == 0:
        change_type = "unchanged"
    elif removed / total < PURE_THRESHOLD:
        change_type = "addition"
    elif added / total < PURE_THRESHOLD:
        change_type = "deletion"
    else:
        change_type = "modification"

    return {"type": change_type, "tokens_added": added, "tokens_removed": removed}


def collect(results: list[dict]) -> list[dict]:
    """One row per evaluated section draft, with change type and scores."""
    rows = []
    for r in results:
        if "error" in r:
            continue
        for d in r.get("section_drafts", []):
            ev = d.get("eval", {})
            if not ev or "skipped" in ev:
                continue
            change = classify_change(d.get("before", ""), d.get("after", ""))
            scores = ev.get("judge", {}).get("scores", {})
            rows.append({
                "ndaa": f"{r['ndaa_year']} s{r['ndaa_section']}",
                "section": d.get("section", "?"),
                "role": d.get("role", "?"),
                **change,
                "bleu": ev.get("bleu"),
                **{k: scores.get(k) for k in JUDGE_KEYS},
            })
    return rows


def _avg(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def _fmt(value: float | None, width: int = 6, prec: int = 2) -> str:
    return f"{value:>{width}.{prec}f}" if value is not None else " " * (width - 1) + "-"


def print_split(rows: list[dict]) -> None:
    metrics = ["bleu"] + JUDGE_KEYS
    header = f"  {'Change type':<14} {'n':>4}" + "".join(
        f" {m[:12]:>12}" for m in metrics)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ct in CHANGE_TYPES + ["ALL"]:
        group = rows if ct == "ALL" else [r for r in rows if r["type"] == ct]
        if not group:
            continue
        cells = "".join(
            f" {_fmt(_avg([r[m] for r in group]), width=12, prec=1 if m == 'bleu' else 2)}"
            for m in metrics)
        print(f"  {ct:<14} {len(group):>4}{cells}")


def _split_table(rows: list[dict]) -> tuple[list[str], list[str], list[list]]:
    """Build the per-change-type averages table.

    Returns (row_labels, col_labels, cells) where cells[i][j] is the average
    of metric j over change type i (None where the type has no sections).
    """
    metrics = ["bleu"] + JUDGE_KEYS
    row_labels, cells = [], []
    for ct in CHANGE_TYPES + ["ALL"]:
        group = rows if ct == "ALL" else [r for r in rows if r["type"] == ct]
        if not group:
            continue
        row_labels.append(f"{ct} (n={len(group)})")
        cells.append([_avg([r[m] for r in group]) for m in metrics])
    return row_labels, metrics, cells


def plot_split(rows: list[dict], out_path: str | Path,
               title: str | None = None) -> Path:
    """Render the per-change-type score table as a figure.

    Judge columns (1-5) and BLEU (0-100) are shaded on independent green
    scales so higher scores read darker. Saved to out_path (PNG/PDF/SVG by
    extension). Returns the path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    row_labels, metrics, cells = _split_table(rows)
    if not row_labels:
        raise ValueError("No rows to plot.")

    col_labels = ["BLEU"] + [k.replace("_", "\n") for k in JUDGE_KEYS]
    # Per-column normalization: BLEU on 0-100, judges on 1-5.
    norms = [colors.Normalize(0, 100)] + [colors.Normalize(1, 5)] * len(JUDGE_KEYS)
    cmap = plt.get_cmap("Greens")

    text, cell_colors = [], []
    for row in cells:
        text_row, color_row = [], []
        for j, val in enumerate(row):
            if val is None:
                text_row.append("-")
                color_row.append("white")
            else:
                prec = 1 if metrics[j] == "bleu" else 2
                text_row.append(f"{val:.{prec}f}")
                # Light tint so black text stays readable.
                rgba = cmap(0.15 + 0.5 * norms[j](val))
                color_row.append(rgba)
        text.append(text_row)
        cell_colors.append(color_row)

    fig, ax = plt.subplots(
        figsize=(1.4 * len(col_labels) + 2, 0.6 * len(row_labels) + 1.2))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0 or c == -1:  # header row / row labels
            cell.set_text_props(fontweight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_per_section(rows: list[dict]) -> None:
    print(f"\n  {'NDAA':<18} {'Section':<28} {'Type':<13} "
          f"{'+tok':>6} {'-tok':>6} {'BLEU':>6} {'Judge':>6}")
    print("  " + "-" * 90)
    for r in sorted(rows, key=lambda r: (r["type"], r["ndaa"], r["section"])):
        print(f"  {r['ndaa']:<18} {r['section'][:28]:<28} {r['type']:<13} "
              f"{r['tokens_added']:>6} {r['tokens_removed']:>6} "
              f"{_fmt(r['bleu'], prec=1)} {_fmt(r['overall'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split eval scores by ground-truth change type")
    parser.add_argument("--input", type=str, default=None,
                        help="Eval results JSON (default: latest "
                             "eval_1n_results*.json in data/results)")
    parser.add_argument("--per-section", action="store_true",
                        help="Also list every section with its change type")
    parser.add_argument("--plot", type=str, nargs="?", const="",
                        help="Render the per-change-type table to an image "
                             "(default: <input>_by_change_type.png)")
    args = parser.parse_args()

    input_path = args.input or str(latest_eval_file())
    print(f"Loading eval results from {input_path} ...")
    with open(input_path) as f:
        results = json.load(f)

    rows = collect(results)
    if not rows:
        print("No evaluated sections found.")
        return

    print(f"  {len(rows)} evaluated sections\n")
    print_split(rows)

    if args.per_section:
        print_per_section(rows)

    if args.plot is not None:
        plot_path = args.plot or str(
            Path(input_path).with_name(
                Path(input_path).stem + "_by_change_type.png"))
        out = plot_split(rows, plot_path, title="Scores by change type")
        print(f"\n  Plot saved to {out}")


if __name__ == "__main__":
    main()
