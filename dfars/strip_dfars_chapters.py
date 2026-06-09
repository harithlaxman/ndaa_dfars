"""Split Title 48 eCFR XML files into per-chapter subsets, without touching the
originals.

Each title-48 XML wraps every chapter as a flat <DIV3 N="..." TYPE="CHAPTER">
sibling under <DIV1>. We keep the file header/wrapper, retain only the requested
chapter's DIV3 block, and re-close the outer <DIV1>/<ECFR> tags.

  Chapter 1 -> FAR   (parts 1-99)
  Chapter 2 -> DFARS (parts 200-299)

Outputs go to data/FAR/ and data/DFARS/, reusing the original filenames.
"""

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "ecfr"
OUT_ROOT = _PROJECT_ROOT / "data" / "DFARS"

# chapter number -> output subdirectory name
CHAPTERS = {"1": "FAR", "2": "DFARS"}

FIRST_DIV3 = re.compile(r"<DIV3 ")


def extract_chapter(text: str, chapter: str, name: str) -> str:
    # Prefix: everything before the first chapter (keeps <?xml?>, <ECFR>,
    # <DIV1>, and the title <HEAD>; drops every chapter body).
    first = FIRST_DIV3.search(text)
    if not first:
        raise ValueError(f"{name}: no <DIV3> found")
    prefix = text[: first.start()]

    open_tag = f'<DIV3 N="{chapter}" TYPE="CHAPTER">'
    start = text.find(open_tag)
    if start == -1:
        raise ValueError(f"{name}: chapter {chapter} not found")

    # Chapters are not nested, so the first </DIV3> after the opening closes it.
    end_tag = "</DIV3>"
    end = text.find(end_tag, start)
    if end == -1:
        raise ValueError(f"{name}: no closing </DIV3> for chapter {chapter}")
    end += len(end_tag)

    return prefix + text[start:end] + "</DIV1>\n</ECFR>\n"


def main(argv):
    dry_run = "--apply" not in argv

    files = sorted(DATA_DIR.glob("title-48_*.xml"))
    if not files:
        print("No title-48 XML files found", file=sys.stderr)
        return 1

    if not dry_run:
        for sub in CHAPTERS.values():
            (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    for path in files:
        text = path.read_text(encoding="utf-8")
        for chapter, sub in CHAPTERS.items():
            out = extract_chapter(text, chapter, path.name)
            pct = 100 * len(out) / len(text)
            print(f"[{sub:5}] {path.name}: {len(text):>10,} -> {len(out):>9,} bytes ({pct:4.1f}%)")
            if not dry_run:
                (OUT_ROOT / sub / path.name).write_text(out, encoding="utf-8")

    if dry_run:
        print("\nDry run only. Re-run with --apply to write data/FAR and data/DFARS.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
