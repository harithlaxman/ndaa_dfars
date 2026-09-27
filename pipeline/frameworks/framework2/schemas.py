"""
Pydantic schemas for structured LLM output.

These models define the structured responses expected from the LLM
at various stages of the 1:N NDAA -> DFARS drafting pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeAssignment(BaseModel):
    """The changes routed to a single DFARS node."""
    node_index: int = Field(
        description="Index of the DFARS node, exactly as shown in the input list."
    )
    change_numbers: list[int] = Field(
        default_factory=list,
        description="Numbers (from the numbered change list) of the changes this "
        "node must implement. Empty if no change is routed to this node.",
    )


class DelegationPlan(BaseModel):
    """Routing plan: which numbered changes each DFARS node must implement."""
    assignments: list[NodeAssignment] = Field(default_factory=list)


class SectionDraft(BaseModel):
    """The revised text for a single DFARS section."""
    revised_text: str = Field(
        description="The complete text of the section after applying the "
        "proposed changes, preserving any existing text the NDAA does not "
        "require changing. Regulatory text only -- no diff tags, no commentary."
    )


class SectionCorrection(BaseModel):
    """Cross-section corrections to apply to one DFARS section's draft."""
    section: str = Field(
        description="The DFARS section number, exactly as shown in the input."
    )
    corrections: list[str] = Field(
        default_factory=list,
        description="Specific, minimal corrections this section's draft needs for "
        "cross-section consistency. Empty if the draft is already consistent.",
    )


class ReconcilePlan(BaseModel):
    """Per-section correction directive from the reconciliation review step.

    The coordinator stage of per-section reconciliation: it identifies cross-
    section issues and routes the specific corrections each section needs, but
    drafts nothing. A separate per-section step applies the corrections.
    """
    corrections: list[SectionCorrection] = Field(default_factory=list)
    issues: str = Field(
        default="",
        description="Brief summary of the cross-section issues found.",
    )
