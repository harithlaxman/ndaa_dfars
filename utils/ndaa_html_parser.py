#!/usr/bin/env python3
"""Parse the govinfo.gov plaw HTML for an NDAA into individual sections.

The govinfo ``link/plaw`` API returns a public law as
``<html><body><pre>...</pre></body></html>`` — the whole statute as one
preformatted text blob. ``parse_plaw_html`` splits that blob into sections,
attaching the enclosing TITLE and Subtitle as metadata.

Format facts (verified against PLAW-116publ283 / FY2021):
  - Body section headings start at column 0 with uppercase ``SEC. <n>.`` (the
    very first section is the literal ``SECTION 1.``). A heading may wrap across
    several lines; it ends at the first blank line.
  - Table-of-contents entries use mixed-case ``Sec. <n>.`` — the case is what
    distinguishes a TOC listing from a real body section. The law has one
    front-matter TOC plus a mini-TOC per division/title.
  - ``TITLE <roman>--``, ``Subtitle <A-Z>--`` and ``DIVISION <A-Z>--`` headers
    are centered (leading whitespace) and appear in BOTH the TOC and the body.
  - ``[[Page 134 STAT. nnnn]]`` markers are sprinkled throughout, standalone and
    surrounded by blank lines; they can fall mid-heading.
  - ``<<NOTE: ...>>`` editorial/USC-citation markers appear inline in headings
    and body text.

Usage:
    from ndaa.parse_ndaa_html import parse_plaw_html
    sections = parse_plaw_html(html, 2021)
"""

import html
import re

# ─── Line patterns ─────────────────────────────────────────────────────────────

# Body section heading: column 0, uppercase "SEC. 101." or "SECTION 1.".
_SECTION_RE = re.compile(r"^(?:SEC\.|SECTION)\s+(\d+[A-Z]?)\.\s*(.*)$")
# Table-of-contents listing: mixed-case "Sec. 101." at column 0.
_TOC_RE = re.compile(r"^Sec\.\s+\d")
# Centered structural headers (leading whitespace).
_TITLE_RE = re.compile(r"^\s+TITLE\s+([IVXLCDM]+)--", re.IGNORECASE)
_SUBTITLE_RE = re.compile(r"^\s+Subtitle\s+([A-Z])--")
_DIVISION_RE = re.compile(r"^\s+DIVISION\s+([A-Z])--")
# Page marker, e.g. "[[Page 134 STAT. 3389]]".
_PAGE_RE = re.compile(r"^\s*\[\[Page .*?\]\]\s*$")
# USC citation inside a <<NOTE: ...>> marker, e.g. "15 USC 9411 note".
_USC_RE = re.compile(r"\d+\s+U\.?\s?S\.?\s?C\.?\s+\d+[A-Za-z0-9-]*(?:\s+note)?")
# A <<NOTE: ...>> marker (after unescaping).
_NOTE_RE = re.compile(r"<<NOTE:(.*?)>>", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def _extract_pre(text: str) -> str:
    """Return the inner text of the <pre> block, or the whole input if absent."""
    match = re.search(r"<pre>(.*)</pre>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else text


def _strip_page_markers(lines: list[str]) -> list[str]:
    """Drop page-marker lines and a single adjacent blank line.

    Removing the surrounding blank means a page break that fell mid-heading
    leaves the heading's lines contiguous, so heading detection still works.
    """
    out: list[str] = []
    skip_next_blank = False
    for line in lines:
        if _PAGE_RE.match(line):
            # Drop a blank we already emitted just above the marker.
            if out and out[-1].strip() == "":
                out.pop()
            skip_next_blank = True
            continue
        if skip_next_blank:
            skip_next_blank = False
            if line.strip() == "":
                continue
        out.append(line)
    return out


def _usc_notes(text: str) -> list[str]:
    """Pull USC citations out of every <<NOTE: ...>> marker in ``text``."""
    notes: list[str] = []
    for note in _NOTE_RE.findall(text):
        for cite in _USC_RE.findall(note):
            cite = _WS_RE.sub(" ", cite).strip()
            if cite not in notes:
                notes.append(cite)
    return notes


def parse_plaw_html(html_text: str, year: int) -> list[dict]:
    """Parse govinfo plaw HTML into a list of section records.

    Each record is::

        {
          "year":      <int>,
          "title":     <roman numeral str or None>,
          "subtitle":  <uppercase letter str or None>,
          "section":   <section number str, e.g. "101">,
          "heading":   <section heading, wrapped lines joined into one>,
          "text":      <raw body text, page markers removed, entities unescaped>,
          "usc_notes": <list of USC citations found in <<NOTE: ...>> markers>,
        }

    Args:
        html_text: the ``<html><body><pre>...</pre>...`` payload from govinfo.
        year: the NDAA fiscal year, stored on every record.
    """
    body = html.unescape(_extract_pre(html_text))
    lines = _strip_page_markers(body.splitlines())

    sections: list[dict] = []
    title = subtitle = None
    in_toc = False
    cur: dict | None = None          # the section currently being filled
    heading_lines: list[str] = []    # raw heading lines until the blank line
    body_lines: list[str] = []       # raw body lines for the current section
    in_heading = False               # still collecting the multi-line heading

    def flush() -> None:
        """Finalize the current section and append it to ``sections``."""
        nonlocal cur, heading_lines, body_lines, in_heading
        if cur is None:
            return
        heading = _WS_RE.sub(" ", " ".join(heading_lines)).strip()
        text = "\n".join(body_lines).strip("\n")
        cur["heading"] = heading
        cur["text"] = text
        cur["usc_notes"] = _usc_notes(heading + "\n" + text)
        sections.append(cur)
        cur, heading_lines, body_lines, in_heading = None, [], [], False

    for line in lines:
        sec_match = _SECTION_RE.match(line)
        if sec_match:
            number, rest = sec_match.group(1), sec_match.group(2)
            in_toc = False
            # Skip anything before the first TITLE (the act-level SEC. 1-4 that
            # sit above any division/title); we only want titled sections.
            if title is None:
                flush()
                continue
            # Division D repeats a section's "SEC. <n>." line as a table caption.
            # If the number matches the section we just opened, it's not a new
            # section — fold it into the current one's body.
            if cur is not None and cur["section"] == number:
                body_lines.append(line)
                in_heading = False
                continue
            flush()
            cur = {"year": year, "title": title, "subtitle": subtitle,
                   "section": number}
            heading_lines = [rest] if rest else []
            body_lines = []
            in_heading = True
            continue

        if _TOC_RE.match(line):
            in_toc = True
            # A TOC line inside a section's body would otherwise pollute it; once
            # we're in a TOC we stop appending to the current section.
            if cur is not None:
                flush()
            continue

        if _DIVISION_RE.match(line):
            # A new division resets the title/subtitle context. The division
            # letter itself is not emitted (NDAA section numbers don't collide
            # across divisions, so (year, section) is unique without it).
            if not in_toc:
                title = subtitle = None
                flush()
            in_heading = False
            continue

        title_match = _TITLE_RE.match(line)
        if title_match:
            if not in_toc:
                title = title_match.group(1).upper()
                subtitle = None
                flush()
            in_heading = False
            continue

        sub_match = _SUBTITLE_RE.match(line)
        if sub_match:
            if not in_toc:
                subtitle = sub_match.group(1)
                flush()
            in_heading = False
            continue

        # Plain content line.
        if cur is None:
            continue
        if in_heading:
            if line.strip() == "":
                in_heading = False  # blank line ends the heading
            else:
                heading_lines.append(line)
        else:
            body_lines.append(line)

    flush()
    return sections
