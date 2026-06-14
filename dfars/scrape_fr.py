import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.api_utils import get_sections

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = str(_PROJECT_ROOT / "data" / "tracker.csv")
OUTPUT_CSV = str(_PROJECT_ROOT / "data" / "all_frcases.csv")
SINGLE_NDAA_OUTPUT_CSV = str(_PROJECT_ROOT / "data" / "single_ndaa_frcases.csv")


def scrape_fr():
    tracker = pd.read_csv(INPUT_CSV)

    # Cache fetched FR documents keyed by (citation, date) so the same FR
    # document is not re-fetched for every NDAA section that cites it.
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
                fetch_cache[key] = get_sections(citation, date)
            if fetch_cache[key]:
                docs.append(fetch_cache[key])

        # Skip sections whose FR documents all failed to fetch / parse.
        if not docs:
            continue

        # Skip sections that do not affect any CFR parts.
        if not any(d.get("implements") for d in docs):
            continue

        # Aggregate the FR fields across all citations for this NDAA section.

        rows.append({
            "ndaa_year": row.get("ndaa_year"),
            "ndaa_section": row.get("ndaa_section"),
            "case_number": row.get("case_number"),
            "citation": row.get("citation"),
            "publication_date": row.get("publication_date"),
            "effective_on": ";".join(d.get("effective_on") or "" for d in docs),
            "cfr_parts": ";".join(
                ",".join(str(p) for p in sorted(d["implements"].keys(), key=int)) if d.get("implements") else ""
                for d in docs
            ),
            "sections": ";".join(
                ",".join(sec for part_secs in d["implements"].values() for sec in part_secs) if d.get("implements") else ""
                for d in docs
            ),
            "body_html_url": ";".join(d.get("body_html_url") or "" for d in docs),
        })

    if rows:
        columns = [
            "ndaa_year",
            "ndaa_section",
            "case_number",
            "citation",
            "publication_date",
            "effective_on",
            "cfr_parts",
            "sections",
            "body_html_url",
        ]
        df_out = pd.DataFrame(rows)[columns]
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully built {len(df_out)} NDAA section rows.")
        print(f"Saved to {OUTPUT_CSV}")

        # NDAA identifier: YEAR_SECTION.
        df_out["_id"] = (
            df_out["ndaa_year"].astype(str) + "_" + df_out["ndaa_section"].astype(str)
        )

        # Map each case to the set of unique NDAA sections it implements.
        case_to_ndaa = {}
        for _, r in df_out.iterrows():
            case_to_ndaa.setdefault(r["case_number"], set()).add(r["_id"])

        # Keep only cases that implement exactly one NDAA section.
        single_ndaa_cases = {
            case for case, ndaas in case_to_ndaa.items() if len(ndaas) == 1
        }

        single_columns = [
            "_id",
            "cfr_parts",
            "sections",
            "publication_date",
            "effective_on",
            "case_number",
            "citation",
            "body_html_url",
        ]
        df_single = df_out[df_out["case_number"].isin(single_ndaa_cases)][single_columns]
        df_single.to_csv(SINGLE_NDAA_OUTPUT_CSV, index=False)
        print(f"\nFound {len(single_ndaa_cases)} cases implementing a single NDAA section.")
        print(f"Saved {len(df_single)} rows to {SINGLE_NDAA_OUTPUT_CSV}")
    else:
        print("\nNo documents fetched.")

if __name__ == "__main__":
    scrape_fr()
