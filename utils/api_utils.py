#!/usr/bin/env python3
"""Helpers for pulling regulatory metadata from public APIs.

Currently covers the Federal Register API (federalregister.gov); GovInfo
(govinfo.gov) helpers will be added below.

"""

import re
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
    """Return the FR final rule for a citation + date, with its affected CFR sections grouped by part.

    Only includes parts and sections where the part lies between 200 and 299 (DFARS range).

    Args:
        citation: a Federal Register citation, e.g. "89 FR 53502".
        fr_date: the rule's publication date, MM/DD/YY or MM/DD/YYYY, e.g. "06/27/24".

    Returns:
        A dict with ``document_number``, ``citation``, ``title``, ``body_html_url``,
        ``effective_on``, ``cfr_references`` (list of parts), and ``implements`` (a dict of
        {part: [sections]} matching the 200-299 range), or ``None`` if the citation or date
        is unparseable, no matching rule is found, or the request/HTML fetch fails.
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
            "effective_on",
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
                    try:
                        part_int = int(part)
                        if 200 <= part_int <= 299:
                            parts.append(part)
                    except ValueError:
                        pass

            body_html_url = doc.get("body_html_url")
            grouped = {}
            if body_html_url:
                html_text = _fetch_html(body_html_url)
                if html_text is not None:
                    sections_list = _parse_sections(html_text)
                    for sec in sections_list:
                        section_num = sec.get("section", "").strip()
                        if not section_num:
                            continue
                        # Inferred part: first 3 digits of the section number
                        match = re.search(r'^\d{3}', section_num)
                        if not match:
                            continue
                        part = match.group(0)
                        try:
                            part_int = int(part)
                            if 200 <= part_int <= 299:
                                if part not in grouped:
                                    grouped[part] = []
                                if section_num not in grouped[part]:
                                    grouped[part].append(section_num)
                        except ValueError:
                            continue

            return {
                "document_number": doc.get("document_number"),
                "citation": doc.get("citation"),
                "title": doc.get("title"),
                "body_html_url": body_html_url,
                "effective_on": doc.get("effective_on"),
                "cfr_references": grouped.keys(),
                "implements": grouped,
            }

    print(f"  No page match for {citation} on {iso_date}")
    return None


# ─── GovInfo link service ──────────────────────────────────────────────────────────
# Thin wrappers over the GovInfo link service (https://www.govinfo.gov/link-docs/).
# The link service maps stable, human-meaningful identifiers (a public law number,
# a U.S. Code citation, a Statutes at Large citation) onto the canonical govinfo.gov
# document for that identifier. We request ``link-type=html`` and return the document's
# extracted plain text, so these helpers can be used directly as LLM tools. A miss is
# answered by the service with HTTP 400 and surfaces here as ``None``.
#
# Reference: https://github.com/usgpo/link-service

GOVINFO_LINK_URL = "https://www.govinfo.gov/link"


def _extract_body_text(html_text):
    """Extract readable plain text from a GovInfo document's HTML body.

    GovInfo serves laws/statutes as ``<pre>``-formatted text and U.S. Code sections as
    styled block elements; ``separator="\\n"`` handles both, and runs of blank lines
    are collapsed. Returns ``None`` if the body is empty.
    """
    tree = HTMLParser(html_text)
    for node in tree.css("head"):
        node.decompose()
    body = tree.css_first("body")
    if body is None:
        return None

    text = body.text(separator="\n", strip=False)
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned or None


def _govinfo_text(path, **extra_params):
    """Fetch a link service document as HTML and return its extracted plain text.

    Requests ``link-type=html`` and follows the service's redirect to the document.

    Args:
        path: the link path after ``/link/``, e.g. ``"uscode/10/2304"``.
        **extra_params: additional query parameters (e.g. ``type``, ``year`` for USC);
            keys whose value is ``None`` are dropped.

    Returns:
        The document's plain text, or ``None`` when nothing matches (the link service
        answers a miss with HTTP 400), the document has no HTML rendition, or the
        request fails.
    """
    params = {k: v for k, v in extra_params.items() if v is not None}
    params["link-type"] = "html"

    url = f"{GOVINFO_LINK_URL}/{path}"
    try:
        response = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  GovInfo fetch error for {url}: {e}")
        return None

    # GovInfo serves UTF-8 but sends no charset, so requests defaults to ISO-8859-1.
    response.encoding = "utf-8"
    return _extract_body_text(response.text)


# ─── GovInfo: Public and Private Laws (PLAW) ───────────────────────────────────────


def get_public_law(congress, lawnum, lawtype="public"):
    """Return the text of a public or private law by Congress, law type, and number.

    Args:
        congress: Congress number, e.g. ``111``.
        lawnum: law number within that Congress and type, e.g. ``78``.
        lawtype: ``"public"`` (default) or ``"private"``.

    Returns:
        The law's plain text, or ``None`` if it can't be found or fetched.
        Example: ``get_public_law(111, 78)``.
    """
    return _govinfo_text(f"plaw/{congress}/{lawtype}/{lawnum}")


def get_law_by_bill(congress, bill_number):
    """Return the text of the law enacted from a given bill (if one exists).

    Args:
        congress: Congress number, e.g. ``111``.
        bill_number: the associated primary bill number printed at the head of the
            law, e.g. ``"S. 3397"`` or ``"H.R. 2544"`` (whitespace is stripped and the
            value lower-cased, so ``"s.3397"`` works too).

    Returns:
        The law's plain text, or ``None`` if it can't be found or fetched.
    """
    bill = "".join(str(bill_number).split()).lower()
    return _govinfo_text(f"plaw/{congress}/{bill}")


def get_law_by_statute_citation(citation):
    """Return the text of a law identified by its Statutes at Large citation.

    Args:
        citation: a Statutes at Large citation as printed atop each page of a law,
            e.g. ``"124 Stat 2859"`` or ``"124 stat 2859"``. Spaces are replaced with
            ``+`` and the value lower-cased, as the link service expects.

    Returns:
        The law's plain text, or ``None`` if it can't be found or fetched.
    """
    cite = "+".join(str(citation).lower().split())
    return _govinfo_text(f"plaw/{cite}")


# ─── GovInfo: United States Code (USCODE) ──────────────────────────────────────────


def get_usc_section(title, section, usc_type="usc", year=None):
    """Return the text of a section of the United States Code.

    Args:
        title: U.S. Code title number, e.g. ``10``.
        section: section number, e.g. ``2304`` (may include a hyphen, e.g.
            ``"2403-1"``).
        usc_type: ``"usc"`` (default) for the main Code or ``"uscappendix"`` for an
            appendix section.
        year: four-digit edition year (e.g. ``2011``) or ``"mostrecent"``. When None,
            the link service returns the most recent edition.

    Returns:
        The section's plain text, or ``None`` if it can't be found or fetched.
        Example: ``get_usc_section(10, 2304)``.
    """
    return _govinfo_text(
        f"uscode/{title}/{section}",
        type=usc_type if usc_type != "usc" else None,
        year=year,
    )


# ─── GovInfo: Statutes at Large (STATUTE) ──────────────────────────────────────────


def get_statute_by_law(congress, lawnum, lawtype="public"):
    """Return the text of a Statutes at Large document by Congress, law type, and number.

    Args:
        congress: Congress number, e.g. ``108``.
        lawnum: law number within that Congress and type, e.g. ``481``.
        lawtype: ``"public"`` (default) or ``"private"``.

    Returns:
        The document's plain text, or ``None`` if it can't be found or fetched.
    """
    return _govinfo_text(f"statute/{congress}/{lawtype}/{lawnum}")


def get_statute_by_citation(volume, page):
    """Return the text of a Statutes at Large document by volume and page.

    Note: when a page holds multiple granules, the HTML rendition returns only the last
    granule on the page (a GovInfo limitation).

    Args:
        volume: Statutes at Large volume number, e.g. ``118``.
        page: page number within the volume, e.g. ``3910``.

    Returns:
        The document's plain text, or ``None`` if it can't be found or fetched.
    """
    return _govinfo_text(f"statute/{volume}/{page}")
