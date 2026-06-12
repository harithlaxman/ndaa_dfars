"""Ingest the extracted DFARS versions into MongoDB, one doc per node per version.

For every `data/DFARS/title-48_<date>.json` hierarchy graph, join on its matching
`title-48_<date>.xml` (for each node's own text) and `usc_citations_<date>.json`
(for each node's USC citations), and build one document per node:

    {
      "_id": "252.204-7012_2024-09-25",   # <section_number>_<version_date>
      "doc_type": "DFARS",
      "section_number": "252.204-7012",
      "version_date": datetime(2024, 9, 25),
      "section": {"number", "type", "parent", "children", "text"},
      "extracted_citations": {"usc": [...]}
    }

The hierarchy graph is the authoritative node set; text and citations are looked
up onto it (missing → "" / []). The collection is dropped and reloaded with a
single bulk insert, then indexed.

Usage:
    uv run dfars/ingest_dfars.py            # all versions
    uv run dfars/ingest_dfars.py --limit 3  # first 3 versions (dry slice)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dfars.extract_usc_citations import node_texts
from utils.mongo_utils import getMongoClient, insert_docs, create_dfars_indexes

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "DFARS"

DB_NAME = "ndaa_dfars"
COLLECTION_NAME = "dfars"


def build_docs(date: str) -> list[dict]:
    """Build the per-node documents for a single version date."""
    graph = json.loads((DATA_DIR / f"title-48_{date}.json").read_text())
    texts = node_texts(DATA_DIR / f"title-48_{date}.xml")

    citations_path = DATA_DIR / f"usc_citations_{date}.json"
    section_to_citations = (
        json.loads(citations_path.read_text()).get("section_to_citations", {})
        if citations_path.exists()
        else {}
    )

    version_date = datetime.strptime(date, "%Y-%m-%d")

    docs = []
    for number, node in graph.items():
        docs.append({
            "_id": f"{number}_{date}",
            "doc_type": "DFARS",
            "section_number": number,
            "version_date": version_date,
            "section": {
                "number": number,
                "type": node["type"],
                "parent": node["parent"],
                "children": node["children"],
                "text": texts.get(number, ""),
            },
            "extracted_citations": {
                "usc": section_to_citations.get(number, []),
            },
        })
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest DFARS versions into MongoDB.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only ingest the first N versions (oldest first).")
    args = parser.parse_args()

    hierarchy_files = sorted(DATA_DIR.glob("title-48_*.json"))
    dates = [p.stem.replace("title-48_", "") for p in hierarchy_files]
    if args.limit is not None:
        dates = dates[:args.limit]

    # Build every doc up front, then load the collection in a single bulk insert.
    all_docs = []
    for date in tqdm(dates, desc="Building DFARS docs"):
        all_docs.extend(build_docs(date))

    client = getMongoClient()
    client[DB_NAME][COLLECTION_NAME].drop()
    insert_docs(client, DB_NAME, COLLECTION_NAME, all_docs)
    create_dfars_indexes(client, DB_NAME, COLLECTION_NAME)

    print(f"\nInserted {len(all_docs)} node docs across {len(dates)} versions "
          f"into {DB_NAME}.{COLLECTION_NAME}.")


if __name__ == "__main__":
    main()
