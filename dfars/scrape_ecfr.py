"""
Download DFARS (Title 48, chapter 2) eCFR XML snapshots for every DFARS Final Rule
in fr_cases.csv, picking an accurate after/before snapshot pair per rule for diffing.

Only chapter 2 is fetched (?chapter=2): it is the entire DFARS, downloads in seconds
(~4 MB vs the full title, which currently 504-times out), and lands directly in
data/DFARS/ — the same layout extract_hierarchy.py / ingest_dfars.py read — so the
strip_dfars_chapters.py step is not needed for DFARS.

eCFR records each amendment on the rule's EFFECTIVE date, which can differ slightly
from the effective_on date in fr_cases, so for each FR case we query the eCFR
versions API for every affected CFR part to find the date its amendments actually
landed:

  - per part, take its earliest version date on or after the rule's effective_on date
    (parts with no version in range are dropped);
  - after  = the latest of those per-part earliest dates;
  - before = one day before the earliest of those per-part earliest dates.

Also writes data/DFARS/dfars_diffs.csv mapping each fr_cases row to its before/
after snapshot files.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = str(_PROJECT_ROOT / "data" / "fr_cases.csv")
OUTPUT_DIR = _PROJECT_ROOT / "data"
MANIFEST_CSV = OUTPUT_DIR / "dfars_diffs.csv"

TITLE = "48"
CHAPTER = "2"  # DFARS is chapter 2 of Title 48.
BASE_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
VERSIONS_URL = "https://www.ecfr.gov/api/versioner/v1/versions/title-{title}.json"


def parse_fr_date(date_str):
    """Parse YYYY-MM-DD (or MM/DD/YY, MM/DD/YYYY) into a date, or None if unparseable."""
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def snapshot_path(date):
    """Path to the Title-48 snapshot file for a given date."""
    return OUTPUT_DIR / f"title-{TITLE}_{date.isoformat()}.xml"


def fetch_version_dates(part, cache):
    """Sorted list of amendment dates for one DFARS `part`, memoized in `cache`.

    Queries the eCFR versions API for chapter 2, `part`, and returns the distinct
    ``content_versions[].date`` values as sorted ``date`` objects ([] on failure).
    """
    if part in cache:
        return cache[part]

    url = VERSIONS_URL.format(title=TITLE)
    params = {"chapter": CHAPTER, "part": part}
    dates = []
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            versions = resp.json().get("content_versions", [])
            dates = sorted({
                datetime.strptime(v["date"], "%Y-%m-%d").date()
                for v in versions if v.get("date")
            })
            break
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == 1:
                tqdm.write(f"  Failed to fetch versions for part {part}: {e}")

    cache[part] = dates
    return dates


def select_snapshot_dates(eff_date, parts, cache):
    """Pick (before, after) snapshot dates for a rule's affected `parts`.

    For each part, take its earliest version date on or after `eff_date`; parts with
    no such version are dropped (with a warning). Then after = the latest of those
    per-part earliest dates, before = one day before the earliest of them. Returns
    (None, None) if no part has a version on or after `eff_date`.
    """
    part_mins = []
    for part in parts:
        later = [d for d in fetch_version_dates(part, cache) if d >= eff_date]
        if later:
            part_mins.append(min(later))
        else:
            tqdm.write(f"  Part {part}: no eCFR version on or after "
                       f"{eff_date.isoformat()}; dropping.")

    if not part_mins:
        return None, None
    after = max(part_mins)
    before = min(part_mins) - timedelta(days=1)
    return before, after


def fetch_title(date, out_path):
    """Stream the DFARS (Title-48 chapter-2) XML for `date` to `out_path`, with one retry."""
    url = BASE_URL.format(date=date.isoformat(), title=TITLE)
    params = {"chapter": CHAPTER}
    for attempt in range(2):
        try:
            with requests.get(url, params=params, timeout=120, stream=True) as resp:
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


def split_row_cases(row):
    """Split one fr_cases row into per-case (case, eff_date, parts) tuples.

    Rows bundling several cases use ';' to separate effective_on, case_number and
    cfr_parts; within a cfr_parts segment the parts are ','-separated.
    """
    def _cell(name):
        value = row.get(name)
        return "" if pd.isna(value) else str(value)

    dates = [d for d in _cell("effective_on").split(";") if d.strip()]
    ref_groups = _cell("cfr_parts").split(";")
    cases = [c.strip() for c in _cell("case_number").split(";")]
    out = []
    for i, d in enumerate(dates):
        refs = ref_groups[i] if i < len(ref_groups) else (ref_groups[-1] if ref_groups else "")
        parts = sorted({p.strip() for p in refs.split(",") if p.strip()}, key=int)
        case = cases[i] if i < len(cases) else (cases[0] if cases else "")
        out.append((case, d.strip(), parts))
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Build the per-case before/after manifest and the dedup set of dates to fetch.
    # version_cache memoizes the eCFR versions API response per CFR part.
    manifest_rows = []
    dates = set()
    version_cache = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("effective_on")):
            continue
        for case, date_str, parts in split_row_cases(row):
            eff_date = parse_fr_date(date_str)
            if eff_date is None:
                tqdm.write(f"  Could not parse date: {date_str!r}")
                continue
            if not parts:
                tqdm.write(f"  Case {case} ({date_str}): no CFR parts; skipping.")
                continue

            before, after = select_snapshot_dates(eff_date, parts, version_cache)
            if before is None:
                tqdm.write(f"  Case {case} ({date_str}): no eCFR versions for any "
                           f"part {parts}; skipping.")
                continue

            dates.add(before)
            dates.add(after)
            manifest_rows.append({
                "ndaa_year": row.get("ndaa_year"),
                "ndaa_section": row.get("ndaa_section"),
                "case_number": row.get("case_number"),
                "dfars_case": case,
                "citation": row.get("citation"),
                "effective_on": date_str,
                "cfr_parts": ",".join(parts),
                "before_date": before.isoformat(),
                "after_date": after.isoformat(),
                "before_xml": str(snapshot_path(before).relative_to(_PROJECT_ROOT)),
                "after_xml": str(snapshot_path(after).relative_to(_PROJECT_ROOT)),
            })

    print(f"Fetching Title-{TITLE} snapshots for {len(dates)} unique dates "
          f"(per-part effective dates from the eCFR versions API).\n")

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
