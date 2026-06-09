"""
Download full eCFR Title 48 (FAR/DFARS) XML snapshots for every Final Rule
publication date in fr_cases.csv, plus the day before each date, so each rule
has an after/before pair of the regulation text for diffing.

Also writes data/ecfr/dfars_diffs.csv mapping each fr_cases row to its before/
after snapshot files.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = str(_PROJECT_ROOT / "data" / "fr_cases.csv")
OUTPUT_DIR = _PROJECT_ROOT / "data" / "ecfr"
MANIFEST_CSV = OUTPUT_DIR / "dfars_diffs.csv"

TITLE = "48"
BASE_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"


def parse_fr_date(date_str):
    """Parse MM/DD/YY or MM/DD/YYYY into a date, or None if unparseable."""
    date_str = date_str.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def snapshot_path(date):
    """Path to the Title-48 snapshot file for a given date."""
    return OUTPUT_DIR / f"title-{TITLE}_{date.isoformat()}.xml"


def fetch_title(date, out_path):
    """Stream the full Title-48 XML for `date` to `out_path`, with one retry."""
    url = BASE_URL.format(date=date.isoformat(), title=TITLE)
    for attempt in range(2):
        try:
            with requests.get(url, timeout=120, stream=True) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            # Don't leave a partial file behind on failure.
            out_path.unlink(missing_ok=True)
            if attempt == 1:
                tqdm.write(f"  Failed {date.isoformat()}: {e}")
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Build the per-row before/after manifest and the dedup set of dates to fetch.
    manifest_rows = []
    dates = set()
    for _, row in df.iterrows():
        if pd.isna(row.get("publication_date")):
            continue
        for part in str(row["publication_date"]).split(";"):
            after = parse_fr_date(part)
            if after is None:
                tqdm.write(f"  Could not parse date: {part!r}")
                continue
            before = after - timedelta(days=1)
            dates.add(before)
            dates.add(after)
            manifest_rows.append({
                "ndaa_year": row.get("ndaa_year"),
                "ndaa_section": row.get("ndaa_section"),
                "section_title": row.get("section_title"),
                "case_number": row.get("case_number"),
                "dfars_case": row.get("dfars_case"),
                "citation": row.get("citation"),
                "publication_date": row.get("publication_date"),
                "before_date": before.isoformat(),
                "after_date": after.isoformat(),
                "before_xml": str(snapshot_path(before).relative_to(_PROJECT_ROOT)),
                "after_xml": str(snapshot_path(after).relative_to(_PROJECT_ROOT)),
            })

    print(f"Fetching Title-{TITLE} snapshots for {len(dates)} unique dates "
          f"(publication dates and the day before each).\n")

    fetched, skipped, failed = 0, 0, 0
    for date in tqdm(sorted(dates), desc=f"Fetching eCFR Title-{TITLE} snapshots"):
        out_path = snapshot_path(date)
        if out_path.exists():
            skipped += 1
            continue
        if fetch_title(date, out_path):
            fetched += 1
        else:
            failed += 1

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)

    print(f"\nDone. Fetched {fetched}, skipped {skipped} (already present), "
          f"failed {failed}.")
    print(f"Snapshots saved to {OUTPUT_DIR}")
    print(f"Manifest ({len(manifest_rows)} rows) saved to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
