#!/usr/bin/env python3
"""Fetch the content of a DFARS hierarchy node, as plain text, from the local XML.

A node lives in a `data/DFARS/title-48_<date>.json` hierarchy graph (keyed by
number, with `type`/`parent`/`children`). Its textual content lives in the
matching `data/DFARS/title-48_<date>.xml`. Given a version and a node number,
this returns the node's own text plus all of its descendants, concatenated.

Subsections (e.g. `201.105-3`) are *sibling* DIV elements in the XML rather than
nested inside their section, and synthesized RESERVED parents (e.g. `252.203`)
have no XML element at all — so descendants are walked via the JSON graph's
`children`, rendering each node's own (non-nested-DIV) text.

Usage:
    python utils/get_dfars.py 2024-09-25 201.105   # section 201.105 + subsections
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# This module lives in utils/, so the project root is two levels up.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _PROJECT_ROOT / "data" / "DFARS"

_WS = re.compile(r"\s+")


# ─── Loading ────────────────────────────────────────────────────────────────────

def _load_xml(version: str) -> ET.Element:
    path = DATA_DIR / f"title-48_{version}.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"DFARS XML for version {version} not found at {path}."
        )
    return ET.parse(path).getroot()


def _load_graph(version: str) -> dict:
    path = DATA_DIR / f"title-48_{version}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"DFARS hierarchy JSON for version {version} not found at {path}. "
            f"Run `uv run extract_hierarchy.py` first."
        )
    return json.loads(path.read_text())


def _index_by_number(root: ET.Element) -> dict:
    """Map every numbered division element to its `N` attribute for O(1) lookup."""
    return {el.attrib["N"]: el for el in root.iter() if el.get("N")}


# ─── Text extraction ────────────────────────────────────────────────────────────

def _inline_text(elem: ET.Element) -> str:
    """Flatten an element's mixed content (incl. inline tags) into one line."""
    return _WS.sub(" ", "".join(elem.itertext())).strip()


def _render_table(elem: ET.Element) -> str:
    """Best-effort render of a <TABLE>: one line per row, cells tab-joined."""
    lines = []
    for row in elem.iter("TR"):
        cells = [_inline_text(c) for c in row if c.tag in ("TD", "TH")]
        if any(cells):
            lines.append("\t".join(cells))
    return "\n".join(lines)


def _own_text(elem: ET.Element) -> str:
    """Render an element's own text: its HEAD plus direct, non-structural children.

    Nested DIV* children (subparts/sections/subsections) are skipped here; they
    are rendered separately by walking the hierarchy graph so that sibling
    subsections and synthesized parents are handled uniformly.
    """
    lines = []
    head = elem.find("HEAD")
    if head is not None:
        lines.append(_inline_text(head))
    for child in elem:
        if child.tag == "HEAD" or child.tag.startswith("DIV"):
            continue
        text = _render_table(child) if child.tag == "TABLE" else _inline_text(child)
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


# ─── Public API ─────────────────────────────────────────────────────────────────

def get_content(version: str, number: str) -> str:
    """Return the content of a DFARS node and all its descendants as plain text.

    Args:
        version: the dated version, e.g. "2024-09-25" (selects the title-48 files).
        number: the node's number/identifier, e.g. "201.105" or "252.203-7000".

    Raises:
        FileNotFoundError: the version's XML or JSON is not present locally.
        ValueError: `number` is not a node in that version's hierarchy.
    """
    graph = _load_graph(version)
    if number not in graph:
        raise ValueError(f"Node {number} not found in DFARS version {version}.")
    index = _index_by_number(_load_xml(version))

    def render(num: str) -> str:
        parts = []
        elem = index.get(num)
        if elem is not None:
            own = _own_text(elem)
            if own:
                parts.append(own)
        else:
            # Synthesized RESERVED parent — no XML element of its own.
            parts.append(f"{num} [RESERVED]")
        for child in graph[num]["children"]:
            parts.append(render(child))
        return "\n\n".join(part for part in parts if part)

    return render(number).strip() + "\n"
