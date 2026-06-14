"""Structured-output schemas for the baseline drafting framework.

The baseline hands the model the NDAA text plus every affected DFARS node and asks
it to return the full revised text for each node -- no manifest, no per-node edit
list, just the revised node.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SectionDraft(BaseModel):
    """One revised DFARS node."""

    section: str = Field(
        description="DFARS section/clause number, e.g. '236.606' or '252.204-7012'. "
        "Must match one of the section numbers given in the prompt.",
    )
    revised_text: str = Field(
        description="The full text of the section after applying the changes "
        "mandated by the NDAA. Return the complete node text, not just the edits, "
        "preserving any existing text the NDAA does not require changing.",
    )


class BaselineDraft(BaseModel):
    """All revised nodes for one NDAA."""

    sections: list[SectionDraft]
