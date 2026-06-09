"""Build a flat graph of the DFARS title-48 hierarchy from each XML version.

For every `data/DFARS/title-48_<date>.xml` file, write a matching
`data/DFARS/title-48_<date>.json`. The JSON is a flat map keyed by a node's
number (its identifier), where every node has the same shape:

    {
      "<number>": {"type": <PART|SUBPART|SECTION|SUBSECTION>,
                   "parent": <number or null>,
                   "children": [<number>, ...]}
    }

The hierarchy is PART -> SUBPART -> SECTION -> SUBSECTION (recursive). Sections
and subsections are distinguished by their number: a hyphenated number such as
`206.302-3-70` is a subsection of `206.302-3`, itself a subsection of `206.302`.
When a parent section is absent from the XML (e.g. Part 252 clauses like
`252.203-7000`, where `252.203` is not a section) it is synthesized so its
children still have a parent.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "DFARS"


def is_range(number: str) -> bool:
    """True for range headings like `225.7005-225.7008` or `211.275-1-211.275-3`.

    A hyphen immediately followed by a dotted number marks the second endpoint of
    a range; such entries are standalone sections, not subsections.
    """
    return bool(re.search(r"-\d+\.\d", number))


def parent_number(number: str) -> str | None:
    """The parent section number, or None if `number` is a top-level section."""
    if "-" not in number or is_range(number):
        return None
    return number.rsplit("-", 1)[0]


def natural_key(number: str) -> list:
    """Sort key that orders e.g. -1, -2, -70, -7000 numerically, not lexically."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", number)]


def section_type(number: str) -> str:
    return "SECTION" if parent_number(number) is None else "SUBSECTION"


def build_section_tree(section_divs: list[ET.Element]) -> list[dict]:
    """Nest a flat list of SECTION divs into a section -> subsection tree.

    Real sections keep their document order; missing parents are synthesized.
    Returns the list of root section nodes, each `{"number", "subsections"}`.
    """
    nodes: dict[str, dict] = {}
    roots: list[dict] = []

    def ensure_linked(number: str) -> None:
        """Attach `number` to its parent, creating ancestors as needed."""
        parent = parent_number(number)
        if parent is None:
            if nodes[number] not in roots:
                roots.append(nodes[number])
            return
        if parent not in nodes:
            nodes[parent] = {"number": parent, "subsections": []}
            ensure_linked(parent)
        if nodes[number] not in nodes[parent]["subsections"]:
            nodes[parent]["subsections"].append(nodes[number])

    for div in section_divs:
        number = div.attrib["N"]
        nodes[number] = {"number": number, "subsections": []}

    for div in section_divs:
        ensure_linked(div.attrib["N"])

    def sort_children(node: dict) -> None:
        node["subsections"].sort(key=lambda c: natural_key(c["number"]))
        for child in node["subsections"]:
            sort_children(child)

    for root in roots:
        sort_children(root)
    return roots


def add_section_tree(roots: list[dict], parent_id: str, graph: dict) -> list[str]:
    """Flatten a section tree into `graph`, returning the root ids in order."""
    ids = []
    for node in roots:
        number = node["number"]
        child_ids = add_section_tree(node["subsections"], number, graph)
        graph[number] = {
            "type": section_type(number),
            "parent": parent_id,
            "children": child_ids,
        }
        ids.append(number)
    return ids


def owning_subpart(section: str, subpart_ids: list[str]) -> str | None:
    """The subpart a top-level section belongs to, by number prefix.

    Section `206.302` belongs to subpart `206.3`; the char after the subpart
    number must be a digit so `214.2` doesn't falsely claim `214.4xx`. The
    longest matching subpart wins. Returns None for a section that sits directly
    under the PART (no owning subpart).
    """
    best = None
    for sp in subpart_ids:
        if (
            section.startswith(sp)
            and len(section) > len(sp)
            and section[len(sp)].isdigit()
            and (best is None or len(sp) > len(best))
        ):
            best = sp
    return best


def parse_file(path: Path) -> dict:
    root = ET.parse(path).getroot()
    graph: dict[str, dict] = {}
    for part in root.findall('.//*[@TYPE="PART"]'):
        part_id = part.attrib["N"]
        subpart_ids = [sp.attrib["N"] for sp in part.findall('.//*[@TYPE="SUBPART"]')]
        # Build one section tree for the whole PART so each number is a single
        # node even when the source splits a family across a subpart and the
        # part itself (a 2018 data quirk for 214.201).
        roots = build_section_tree(part.findall('.//*[@TYPE="SECTION"]'))

        part_section_ids: list[str] = []
        subpart_children: dict[str, list[str]] = {sp: [] for sp in subpart_ids}
        for node in roots:
            number = node["number"]
            owner = owning_subpart(number, subpart_ids)
            add_section_tree([node], owner or part_id, graph)
            (subpart_children[owner] if owner else part_section_ids).append(number)

        # Sections directly under the PART always precede its subparts.
        child_ids = list(part_section_ids)
        for subpart_id in subpart_ids:
            graph[subpart_id] = {
                "type": "SUBPART",
                "parent": part_id,
                "children": subpart_children[subpart_id],
            }
            child_ids.append(subpart_id)
        graph[part_id] = {"type": "PART", "parent": None, "children": child_ids}
    return graph


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(DATA_DIR.glob("title-48_*.xml"))
    for xml_path in tqdm(xml_files, desc="Building hierarchy graph"):
        graph = parse_file(xml_path)
        out_path = DATA_DIR / f"{xml_path.stem}.json"
        out_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
