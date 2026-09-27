"""Structured-output schemas for the framework1 drafting framework.

Framework1 hands the model the NDAA text plus every affected DFARS node and asks
it, for each node, to return how it changes (added/modified/deleted) and its full
revised text -- no manifest, no per-node edit list.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SectionDraft(BaseModel):
    """One drafted DFARS node and how it changes under the NDAA."""

    section: str = Field(
        description="DFARS section/clause number, e.g. '236.606' or '252.204-7012'. "
        "For a modified or deleted section, match a section number given in the "
        "prompt. For an added section, assign a new number following the FAR/DFARS "
        "Drafting Guide numbering rules.",
    )
    change_type: Literal["added", "modified", "deleted"] = Field(
        description=(
            "How this section changes under the NDAA: "
            "'modified' = edit the existing section in place (this includes adding "
            "one or more new paragraphs to it); "
            "'added' = a brand-new section/subsection/clause, created only because "
            "the mandate cannot be implemented in any existing section; "
            "'deleted' = the existing section is removed in its entirety."
        ),
    )
    revised_text: str = Field(
        description="The full text of the section after applying the changes. For "
        "'modified', return the complete revised section, preserving any existing "
        "text the NDAA does not require changing. For 'added', return the full text "
        "of the new section. For 'deleted', leave this empty.",
    )


class Framework1Draft(BaseModel):
    """All revised nodes for one NDAA."""

    sections: list[SectionDraft]
