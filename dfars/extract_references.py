"""Annotate each DFARS docs snapshot with internal cross-reference edges.

Every `data/DFARS/docs/title-48_<date>.json` file maps `node_id -> node`, where a
node is `{type, parent, children, heading, text, edges}` and each edge is
`{"destination": <node_id>, "type": <TYPE>}`. All DFARS content lives in parts
200-299, so a node's plain-text references to *other* DFARS sections are easy to
spot by regex (e.g. "...as provided in 215.408" inside node 225.1101).

For each node we scan its `text`, and for every reference that resolves to a real
node in the same snapshot we append a `{"destination": ..., "type": "reference"}`
edge to that node, subject to these filters:

  1. The target must be a key in the same file (we only link to nodes that exist).
  2. PGI references are dropped ("see PGI 201.109" is guidance, not a DFARS xref).
  3. No intra-hierarchy self-links: skip the node itself, any ancestor (walk
     `parent` up to the PART) and any descendant (walk `children` recursively).
  4. If the node already has a PRESCRIBES edge to the target, no reference is added.
  5. At most one reference edge per distinct target per node.

Files are rewritten in place and the operation is idempotent: pre-existing
`reference` edges are stripped before re-deriving them, so re-running does not
accumulate duplicates.

Usage:
    python dfars/extract_references.py
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

from dfars.extract_hierarchy import natural_key

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = _PROJECT_ROOT / "data" / "DFARS" / "docs"

# DFARS node numbers: part (2xx) + at least one dot-separated group, optionally
# followed by hyphenated clause/subsection groups (e.g. 225.1101, 201.6,
# 252.227-7013, 204.7303-4).
REF_RE = re.compile(r"\b2\d\d\.\d+(?:-\d+)*\b")

# A match immediately preceded by "PGI" (with optional whitespace) is a pointer to
# the Procedures, Guidance, and Information companion, not a DFARS cross-reference.
PGI_RE = re.compile(r"PGI\s*$")


def _ancestors(node_id: str, nodes: dict) -> set:
    """Walk `parent` links up to the PART node."""
    seen = set()
    cur = nodes.get(node_id, {}).get("parent")
    while cur and cur in nodes and cur not in seen:
        seen.add(cur)
        cur = nodes[cur].get("parent")
    return seen


def _descendants(node_id: str, nodes: dict) -> set:
    """Walk `children` recursively."""
    seen = set()
    stack = list(nodes.get(node_id, {}).get("children", []))
    while stack:
        child = stack.pop()
        if child in seen or child not in nodes:
            continue
        seen.add(child)
        stack.extend(nodes[child].get("children", []))
    return seen


def extract_references(nodes: dict) -> int:
    """Add `reference` edges to every node in a snapshot; return count added."""
    added = 0
    for node_id, node in nodes.items():
        edges = node.get("edges", [])
        # Idempotency: drop previously-derived reference edges before recomputing.
        edges = [e for e in edges if e.get("type") != "reference"]
        prescribes = {
            e["destination"] for e in edges if e.get("type") == "PRESCRIBES"
        }
        excluded = {node_id} | _ancestors(node_id, nodes) | _descendants(node_id, nodes)

        text = node.get("text", "")
        dests = set()
        for m in REF_RE.finditer(text):
            dest = m.group()
            if dest not in nodes:
                continue
            if PGI_RE.search(text, 0, m.start()):
                continue
            if dest in excluded or dest in prescribes:
                continue
            dests.add(dest)

        for dest in sorted(dests, key=natural_key):
            edges.append({"destination": dest, "type": "reference"})
            added += 1

        node["edges"] = edges
    return added


def main() -> None:
    files = sorted(DOCS_DIR.glob("title-48_*.json"))
    total_edges = 0
    for path in tqdm(files, desc="annotating references"):
        nodes = json.loads(path.read_text())
        added = extract_references(nodes)
        path.write_text(json.dumps(nodes, indent=2, ensure_ascii=False))
        total_edges += added
    print(f"Processed {len(files)} files, added {total_edges} reference edges.")


if __name__ == "__main__":
    main()
