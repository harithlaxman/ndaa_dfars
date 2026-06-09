#!/usr/bin/env python3
"""Fetch and format a single NDAA section, as plain text, from the local bill XML.

The bill XMLs (US Congress bill DTD format) live at
``data/ndaa/xmls/ndaa_{year}.xml`` and are downloaded by ``fetch_ndaa_xmls.py``.

Usage:
    python utils/get_ndaa.py 2021 847   # print SEC. 847 of the FY2021 NDAA
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

# This module lives in utils/, so the project root is two levels up.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Existing NDAA XMLs live under data/ndaa/xmls/. (Note: fetch_ndaa_xmls.py
# currently writes to data/ndaa/ root — keep these reconciled if that changes.)
XML_DIR = _PROJECT_ROOT / "data" / "ndaa" / "xmls"

# Block-level elements that carry their own enum/header/text and nest. The
# structural containers (section/title/chapter/...) appear inside <quoted-block>
# amendments that insert whole sections of existing law.
_BLOCK_TAGS = (
    "subsection",
    "paragraph",
    "subparagraph",
    "clause",
    "subclause",
    "item",
    "subitem",
    "section",
    "division",
    "title",
    "subtitle",
    "part",
    "subpart",
    "chapter",
    "subchapter",
)

# Indentation applied per nesting level in the plain-text output.
_INDENT = "    "

_WS = re.compile(r"\s+")


def _normalize(value) -> str:
    """Normalize a section reference for comparison: str, trim, drop trailing dot."""
    return str(value).strip().rstrip(".").strip()


# ─── Inline text flattening ────────────────────────────────────────────────────

def _inline_text(elem) -> str:
    """Flatten an element's mixed content into a single plain-text string.

    Concatenates text and tails, wrapping <quote> in quotation marks. All other
    inline tags (<external-xref>, <bold>, <italic>, <term>, <short-title>,
    <header-in-text>, ...) simply contribute their text.
    """
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        inner = _inline_text(child)
        if child.tag == "quote":
            parts.append(f"“{inner}”")
        else:
            parts.append(inner)
        if child.tail:
            parts.append(child.tail)
    return _WS.sub(" ", "".join(parts)).strip()


def _child_text(elem):
    """Return the flattened text of a direct <text> child, or '' if none."""
    node = elem.find("text")
    return _inline_text(node) if node is not None else ""


def _enum(elem):
    node = elem.find("enum")
    return _normalize(node.text) if node is not None and node.text else ""


def _header(elem):
    node = elem.find("header")
    return _inline_text(node) if node is not None else ""


# ─── Plain-text rendering ──────────────────────────────────────────────────────

def _render_block(elem, depth, lines):
    """Render a block element (subsection/paragraph/...) as an indented line.

    Follows the usual legislative style: "(a) In general.—text" when a header is
    present, otherwise "(1) text".
    """
    indent = _INDENT * depth
    enum = _enum(elem)
    header = _header(elem)
    text = _child_text(elem)

    prefix = f"{enum} " if enum else ""
    if header and text:
        line = f"{indent}{prefix}{header}.—{text}"
    elif header:
        line = f"{indent}{prefix}{header}."
    else:
        line = f"{indent}{prefix}{text}"
    lines.append(line.rstrip())

    _render_children(elem, depth + 1, lines)


def _render_children(elem, depth, lines):
    """Recurse into block / quoted-block / table children of an element."""
    for child in elem:
        tag = child.tag
        if tag in _BLOCK_TAGS:
            _render_block(child, depth, lines)
        elif tag == "quoted-block":
            _render_quoted_block(child, depth, lines)
        elif tag == "table":
            _render_table(child, depth, lines)


def _render_quoted_block(elem, depth, lines):
    """Render an amendment <quoted-block>: its inserted text, indented one level."""
    enum = _enum(elem)
    header = _header(elem)
    text = _child_text(elem)
    # A quoted-block may itself open with an enum/header/text before nested blocks.
    prefix = f"{enum} " if enum else ""
    if header and text:
        lead = f"{prefix}{header}.—{text}"
    elif header:
        lead = f"{prefix}{header}."
    else:
        lead = f"{prefix}{text}"
    lead = lead.strip()
    if lead:
        lines.append(f"{_INDENT * depth}{lead}")

    _render_children(elem, depth + 1, lines)


def _render_table(elem, depth, lines):
    """Best-effort render of a <table>: one line per row, entries tab-joined."""
    indent = _INDENT * depth
    for row in elem.iter("row"):
        cells = [_inline_text(entry) for entry in row.findall("entry")]
        if any(cells):
            lines.append(indent + "\t".join(cells))


def _render_section(sec) -> str:
    enum = _enum(sec)
    header = _header(sec)
    title = f"SEC. {enum}." if enum else "SEC."
    if header:
        title += f" {header}"

    lines = [title, ""]

    # Some sections carry body text directly (no subsections).
    direct = _child_text(sec)
    if direct:
        lines.append(direct)
        lines.append("")

    _render_children(sec, 0, lines)

    # Collapse any accidental triple blank lines.
    out = "\n".join(lines).rstrip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", out)


# ─── Public API ────────────────────────────────────────────────────────────────

def _load_root(year):
    path = XML_DIR / f"ndaa_{year}.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"NDAA {year} XML not found at {path}. "
            f"Run `uv run fetch_ndaa_xmls.py` first."
        )
    return ET.parse(path).getroot()


def _find_section(root, section):
    target = _normalize(section)
    for sec in root.iter("section"):
        if _enum(sec) == target:
            return sec
    raise ValueError(f"Section {section} not found in NDAA XML.")


def get_section(year, section) -> str:
    """Return the given NDAA section as formatted plain text.

    Args:
        year: NDAA fiscal year, e.g. 2021.
        section: section number, e.g. 847 or "847".

    Raises:
        FileNotFoundError: the year's XML is not present locally.
        ValueError: the section is not found in that year's bill.
    """
    root = _load_root(year)
    sec = _find_section(root, section)
    return _render_section(sec)
