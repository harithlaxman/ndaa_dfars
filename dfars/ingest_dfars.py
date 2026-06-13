"""Ingest the extracted DFARS docs snapshots into MongoDB, one doc per node per version.

For every `data/DFARS/docs/title-48_<date>.json` snapshot — a self-contained
`node_id -> {type, parent, children, heading, text, edges, usc_citations}` map — we
stamp each node with its `version_date` (taken from the filename) and build one
document per node:

    {
      "_id": "252.204-7012_2024-09-25",   # <section_number>_<version_date>
      "doc_type": "DFARS",
      "section_number": "252.204-7012",
      "version_date": datetime(2024, 9, 25),
      "hierarchy": {"part", "subpart", "section"},
      "section": {"number", "type", "parent", "children", "text"},
      "amendment_history": {"raw": "[80 FR 51745, ...]", "edits": [{"cite", "date"}]},
      "extracted_citations": {"usc": [...]},
      "edges": [...]
    }

Each docs node already carries its own text, cross-reference/PRESCRIBES `edges`, and
`usc_citations`, so no XML or sidecar citation files are read. The amendment history
(the section's CITA line — Federal Register publication + amendments) is split out of
the node's `text` into `amendment_history`; `edits` is the parsed, ordered list of
{cite, date} entries. The collection is dropped and reloaded with a single bulk
insert per version, then indexed.

Usage:
    uv run dfars/ingest_dfars.py            # all versions
    uv run dfars/ingest_dfars.py --limit 3  # first 3 versions (dry slice)
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dfars.extract_usc_citations import parse_amendment_history
from utils.mongo_utils import getMongoClient, insert_docs, create_dfars_indexes

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "DFARS"
DOCS_DIR = DATA_DIR / "docs"

DB_NAME = "ndaa_dfars"
COLLECTION_NAME = "dfars"

# The CITA amendment-history line is the bracketed run of Federal Register citations
# that the docs text carries inline (e.g. "[80 FR 51745, Aug. 26, 2015, as amended
# ...]"). A node's body may be followed by an Editorial Note, so the bracket is not
# always the very last token — match any bracket containing an "<vol> FR <page>"
# citation. Verified against the XML CITA: exact match for all 1413 nodes, no
# false positives.
CITA_RE = re.compile(r"\[[^\[\]]*?\d+\s+FR\s+\d+[^\[\]]*?\]")


def split_text_and_history(text: str) -> tuple[str, str]:
    """Separate a docs node's body text from its CITA amendment-history line.

    Returns ``(body, raw_history)`` where ``raw_history`` is the joined CITA
    bracket(s) (``""`` if the node has none) and ``body`` is the text with those
    brackets removed. Any non-CITA content (e.g. a trailing Editorial Note) stays
    in ``body``.
    """
    cites = CITA_RE.findall(text)
    if not cites:
        return text.strip(), ""
    body = CITA_RE.sub("", text)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body, " ".join(c.strip() for c in cites)


def node_hierarchy(graph: dict, number: str) -> dict:
    """Resolve which PART, SUBPART, and SECTION a node belongs to.

    Walks up the ``parent`` chain (DFARS nests PART → SUBPART → SECTION → SUBSECTION)
    and records the first ancestor of each level, counting the node itself. Levels that
    don't apply (e.g. a PART has no subpart/section) come back as ``None``.
    """
    levels = {"PART": None, "SUBPART": None, "SECTION": None}
    cur = number
    while cur is not None and cur in graph:
        node = graph[cur]
        ntype = node["type"]
        if ntype in levels and levels[ntype] is None:
            levels[ntype] = cur
        cur = node["parent"]
    return {
        "part": levels["PART"],
        "subpart": levels["SUBPART"],
        "section": levels["SECTION"],
    }


def build_docs(date: str) -> list[dict]:
    """Build the per-node documents for a single version date."""
    graph = json.loads((DOCS_DIR / f"title-48_{date}.json").read_text())
    version_date = datetime.strptime(date, "%Y-%m-%d")

    docs = []
    for number, node in graph.items():
        body, raw_history = split_text_and_history(node.get("text", ""))
        docs.append({
            "_id": f"{number}_{date}",
            "doc_type": "DFARS",
            "section_number": number,
            "version_date": version_date,
            "hierarchy": node_hierarchy(graph, number),
            "section": {
                "number": number,
                "type": node["type"],
                "parent": node["parent"],
                "children": node["children"],
                "text": body,
            },
            "amendment_history": {
                "raw": raw_history,
                "edits": parse_amendment_history(raw_history),
            },
            "extracted_citations": {
                "usc": node.get("usc_citations", []),
            },
            "edges": node.get("edges", []),
        })
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest DFARS versions into MongoDB.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only ingest the first N versions (oldest first).")
    args = parser.parse_args()

    docs_files = sorted(DOCS_DIR.glob("title-48_*.json"))
    dates = [p.stem.replace("title-48_", "") for p in docs_files]
    if args.limit is not None:
        dates = dates[:args.limit]

    client = getMongoClient()
    # Drop once for a clean reload, then bulk-insert each version and index at the end.
    client[DB_NAME][COLLECTION_NAME].drop()

    l = 0
    for date in tqdm(dates, desc="Building DFARS docs"):
        all_docs = build_docs(date)
        l += len(all_docs)
        insert_docs(client, DB_NAME, COLLECTION_NAME, all_docs)

    create_dfars_indexes(client, DB_NAME, COLLECTION_NAME)

    print(f"\nInserted {l} node docs across {len(dates)} versions "
          f"into {DB_NAME}.{COLLECTION_NAME}.")


if __name__ == "__main__":
    main()
