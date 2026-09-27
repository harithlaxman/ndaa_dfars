# Prerequisites
## Data
The following files are required. Everything else can be fetched from publicly available APIs.
1. [Implementation Tracker](https://www.acq.osd.mil/dpap/dars/docs/DFARS_NDAA_Implementation_Tracker_2025-06-06.pdf) - maintained by DARS to keep track of NDAA to DFARS implementations. Format: PDF
2. `ndaa_plaw.csv` Fiscal Year -> Public Law mapping. Used to by this [[Useful APIs#^2eebfd|govinfo api]] to download the HTML of the NDAA for a given year. (I forgot where I got the csv from, but trust me) Format: HTML ^ndaa-plaw-csv
3. [FAR Drafting Guide](https://www.acq.osd.mil/dpap/dars/docs/far_dfars_guide/FAR%20Drafting%20Guide--April%2030,%202011.pdf) - Edited version is used as a system prompt for drafting changes.
## Infra
1. [uv](https://docs.astral.sh/uv/)
2. Mongo DB instance
   - Set `MONGO_CLIENT_URI` in `.env` 
   - If setting up locally: `docker compose up -d`
# Setup
Parse the tracker (by default looks for `tracker.pdf` in project root):
```sh
$ uv sync
$ uv run python parse_tracker.py
```
## NDAA
1. Fetch Public Laws, parse HTML and ingest all the NDAAs to MongoDB.
```sh
$ uv run python ndaa/extract_and_ingest.py
```
2. **Uses OpenAI API calls:** Extract citations from all the NDAAs. 
```sh
# use with --replace to replace existing docs 
$ uv run python ndaa/extract_citations.py 
```
## DFARS
```sh
# scrape FR to get list of DFARS sections per case
$ uv run python dfars/scrape_fr.py 

# scrape the eCFR to get snapshots of DFARS for each case in XML format
$ uv run python dfars/scrape_ecfr.py

# parse the xml into DFARS nodes - subpart, section, subsection
$ uv run python dfars/extract_heirarchy.py

# ingest into mongodb
$ uv run python dfars/ingest_dfars.py

# DFARS before and after indexed by NDAA
$ uv run python dfars/dfars_diff.py
```
