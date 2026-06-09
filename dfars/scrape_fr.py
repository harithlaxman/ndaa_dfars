import re
import requests
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = str(_PROJECT_ROOT / "data" / "tracker.csv")
OUTPUT_CSV = str(_PROJECT_ROOT / "data" / "fr_cases.csv")

def parse_fr_date(date_str):
    """Parse a date string in MM/DD/YY or MM/DD/YYYY format to YYYY-MM-DD."""
    date_str = date_str.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def fetch_citation_by_date_and_page(citation=None, fr_date=None):
    fields = [
        "document_number",
        "citation",
        "start_page",
        "body_html_url",
        "publication_date",
        "title",
        "cfr_references",
    ]

    params = {
        "conditions[agencies][]": ["defense-acquisition-regulations-system", "defense-department"],
        "conditions[type][]": "RULE",
        "conditions[cfr][title]": "48",
        # "conditions[cfr][part]": "200-299",
        "conditions[search_type_id]": "6",
        "fields[]": fields,
        "per_page": "1000",
    }

    # Initialize variables to avoid UnboundLocalError
    page = None
    iso_date = None

    # Parse citation into volume and page
    if citation is not None:
        parts = citation.strip().split(" FR ")
        if len(parts) != 2:
            tqdm.write(f"  Could not parse citation format: {citation}")
        else:
            page = parts[1].strip()

    # Parse date
    if fr_date is not None:
        iso_date = parse_fr_date(fr_date)
        if not iso_date:
            tqdm.write(f"  Could not parse date: {fr_date}")
    
    if page is not None and iso_date is not None:
        params["conditions[publication_date][gte]"] = iso_date
        params["conditions[publication_date][lte]"] = iso_date
    else:
        params["conditions[term]"] = "NDAA"

    try:
        response = requests.get(
            "https://www.federalregister.gov/api/v1/documents.json",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if page is not None:
            for result in data.get("results", []):
                if str(result.get("start_page")) == page:
                    return result
        else:
            return data.get("results", [])

        tqdm.write(f"  No page match for {citation} on {iso_date}")
        return None

    except requests.exceptions.RequestException as e:
        tqdm.write(f"  API error for {citation}: {e}")
        return None


def enrich_doc(doc):
    """Extract the DFARS case from the title and flatten CFR references.

    Returns the enriched doc, or None if no DFARS case can be parsed.
    """
    title = doc.get("title", "")
    match = re.search(r'DFARS Case ([A-Za-z0-9-]+)', title, re.IGNORECASE)
    if not match:
        return None
    doc["dfars_case"] = match.group(1).strip()
    doc["cfr_references"] = [cit["part"] for cit in doc.get("cfr_references") or []]
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
                doc = fetch_citation_by_date_and_page(citation, date)
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
