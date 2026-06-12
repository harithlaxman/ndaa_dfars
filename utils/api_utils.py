#!/usr/bin/env python3
"""Helpers for pulling regulatory metadata from public APIs.

Currently covers the Federal Register API (federalregister.gov); GovInfo
(govinfo.gov) helpers will be added below.

"""

from datetime import datetime

import requests
from selectolax.parser import HTMLParser

# A real User-Agent avoids federalregister.gov's bot blocking.
_HEADERS = {"User-Agent": "ndaa-dfars/1.0 (research; +https://federalregister.gov)"}
_TIMEOUT = 30


# ─── Federal Register ────────────────────────────────────────────────────────────

FR_DOCUMENTS_URL = "https://www.federalregister.gov/api/v1/documents.json"


def _parse_citation_page(citation):
    """Extract the start page from a '<volume> FR <page>' citation, or None."""
    parts = str(citation).strip().split(" FR ")
    if len(parts) != 2:
        print(f"  Could not parse citation format: {citation}")
        return None
    return parts[1].strip()


def _parse_fr_date(date_str):
    """Parse a date in MM/DD/YY or MM/DD/YYYY format to YYYY-MM-DD, or None."""
    date_str = str(date_str).strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    print(f"  Could not parse date: {date_str}")
    return None


def get_cfr_parts(citation, fr_date):
    """Return the FR final rule for a citation + date, with its affected CFR parts.

    The publication date plus a ``type=RULE`` filter narrow the search to the single
    final rule (the same citation/page can otherwise surface multiple documents).

    Args:
        citation: a Federal Register citation, e.g. "89 FR 53502".
        fr_date: the rule's publication date, MM/DD/YY or MM/DD/YYYY, e.g. "06/27/24".

    Returns:
        A dict with ``document_number``, ``citation``, ``title``, ``body_html_url`` and
        ``cfr_references`` (a de-duplicated list of affected part numbers), or ``None``
        if the citation or date is unparseable, no matching rule is found, or the
        request fails.
    """
    page = _parse_citation_page(citation)
    iso_date = _parse_fr_date(fr_date)
    if page is None or iso_date is None:
        return None

    params = {
        "conditions[agencies][]": [
            "defense-acquisition-regulations-system",
            "defense-department",
        ],
        "conditions[type][]": "RULE",
        "conditions[cfr][title]": "48",
        "conditions[publication_date][gte]": iso_date,
        "conditions[publication_date][lte]": iso_date,
        "fields[]": [
            "document_number",
            "citation",
            "start_page",
            "title",
            "body_html_url",
            "cfr_references",
        ],
        "per_page": "1000",
    }

    try:
        response = requests.get(
            FR_DOCUMENTS_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"  API error for {citation}: {e}")
        return None

    for doc in results:
        if str(doc.get("start_page")) == page:
            # Flatten cfr_references to a de-duplicated, order-preserving part list.
            parts = []
            for ref in doc.get("cfr_references") or []:
                part = ref.get("part")
                if part is not None and part not in parts:
                    parts.append(part)
            return {
                "document_number": doc.get("document_number"),
                "citation": doc.get("citation"),
                "title": doc.get("title"),
                "body_html_url": doc.get("body_html_url"),
                "cfr_references": parts,
            }

    print(f"  No page match for {citation} on {iso_date}")
    return None


def _fetch_html(url):
    """Fetch a URL's text with the module headers, or None on a request error."""
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"  HTML fetch error for {url}: {e}")
        return None


def _parse_sections(html_text):
    """Extract amended CFR sections from an FR rule's full-text HTML.

    Each amended section is marked by a ``sectno sectno-reference`` div (holding the
    section number) immediately followed by a ``section-subject`` div (the heading).
    The table-of-contents uses a different class (``sectno-citation`` /
    ``sectno-subject``) and is therefore excluded.
    """
    sections = []
    for ref in HTMLParser(html_text).css("div.sectno-reference"):
        number = ref.text(strip=True)
        subject = ""
        sibling = ref.next
        while sibling is not None:
            classes = sibling.attributes.get("class") or "" if sibling.tag == "div" else ""
            if "section-subject" in classes:
                subject = sibling.text(strip=True)
                break
            sibling = sibling.next
        sections.append({"section": number, "subject": subject})
    return sections


def get_sections(citation, fr_date):
    """Return the CFR sections amended by an FR final rule.

    Resolves the rule via :func:`get_cfr_parts` (citation + date), fetches its
    full-text HTML, and extracts each amended section.

    Args:
        citation: a Federal Register citation, e.g. "89 FR 53502".
        fr_date: the rule's publication date, MM/DD/YY or MM/DD/YYYY, e.g. "06/27/24".

    Returns:
        A list of ``{"section": <number>, "subject": <heading>}`` dicts in document
        order (e.g. ``{"section": "236.606-70", "subject": "Statutory fee limitation."}``),
        or ``None`` if the rule can't be resolved or its HTML can't be fetched.
    """
    doc = get_cfr_parts(citation, fr_date)
    if not doc or not doc.get("body_html_url"):
        print(f"  No HTML available for {citation} on {fr_date}")
        return None

    html_text = _fetch_html(doc["body_html_url"])
    if html_text is None:
        return None

    return _parse_sections(html_text)


# ─── GovInfo ─────────────────────────────────────────────────────────────────────
# (govinfo.gov helpers to be added.)
