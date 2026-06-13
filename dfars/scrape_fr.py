import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.api_utils import get_cfr_parts

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = str(_PROJECT_ROOT / "data" / "tracker.csv")
OUTPUT_CSV = str(_PROJECT_ROOT / "data" / "fr_cases.csv")


def enrich_doc(doc):
    """Extract the DFARS case from the document's title.

    Takes a doc dict from ``get_cfr_parts`` (whose ``cfr_references`` is already a
    flattened part list). Returns the doc with ``dfars_case`` added, or None if no
    DFARS case can be parsed from the title.
    """
    title = doc.get("title", "")
    match = re.search(r'DFARS Case ([A-Za-z0-9-]+)', title, re.IGNORECASE)
    if not match:
        return None
    doc["dfars_case"] = match.group(1).strip()
    return doc


def scrape_fr():
    tracker = pd.read_csv(INPUT_CSV)

    # Cache fetched (and enriched) FR documents keyed by (citation, date) so the
    # same FR document is not re-fetched for every NDAA section that cites it.
    fetch_cache = {}

    rows = []
    # One output row per unique NDAA section (i.e. per tracker row).
    for _, row in tqdm(tracker.iterrows(), total=len(tracker), desc="Fetching FR documents based on tracker"):
        if pd.isna(row.get("citation")) or pd.isna(row.get("publication_date")):
            continue

        citations = str(row["citation"]).split(";")
        dates = str(row["publication_date"]).split(";")

        docs = []
        for citation, date in zip(citations, dates):
            key = (citation.strip(), date.strip())
            if key not in fetch_cache:
                doc = get_cfr_parts(citation, date)
                fetch_cache[key] = enrich_doc(doc) if doc else None
            if fetch_cache[key]:
                docs.append(fetch_cache[key])

        # Skip sections whose FR documents all failed to fetch / parse.
        if not docs:
            continue

        # Aggregate the FR fields across all citations for this NDAA section.
        rows.append({
            "ndaa_year": row.get("ndaa_year"),
            "ndaa_section": row.get("ndaa_section"),
            "section_title": row.get("section_title"),
            "case_number": row.get("case_number"),
            "citation": row.get("citation"),
            "publication_date": row.get("publication_date"),
            "effective_on": ";".join(d["effective_on"] for d in docs),
            "document_number": ";".join(d["document_number"] for d in docs),
            "dfars_case": ";".join(d["dfars_case"] for d in docs),
            "cfr_references": ";".join(
                ",".join(str(p) for p in d["cfr_references"]) for d in docs
            ),
            "title": ";".join(d.get("title", "") for d in docs),
            "body_html_url": ";".join(d.get("body_html_url", "") for d in docs),
        })

    if rows:
        columns = [
            "ndaa_year",
            "ndaa_section",
            "section_title",
            "case_number",
            "citation",
            "publication_date",
            "effective_on",
            "document_number",
            "dfars_case",
            "cfr_references",
            "title",
            "body_html_url",
        ]
        df_out = pd.DataFrame(rows)[columns]
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully built {len(df_out)} NDAA section rows.")
        print(f"Saved to {OUTPUT_CSV}")
    else:
        print("\nNo documents fetched.")

if __name__ == "__main__":
    scrape_fr()
