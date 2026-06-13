"""
Pydantic schemas for structured LLM output.

These models define the structured responses expected from the LLM
at various stages of the 1:N NDAA -> DFARS drafting pipeline.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MandatedChange(BaseModel):
    """A single change mandated by the NDAA."""
    id: str = Field(description="Change identifier, e.g. 'C1', 'C2'")
    type: str = Field(
        description="One of: new_requirement, modified_threshold, new_definition, repeal, amendment"
    )
    description: str = Field(description="What the NDAA requires")
    statutory_basis: str = Field(description="Specific section reference")
    key_terms: list[str] = Field(default_factory=list)
    relevant_text_snippet: str = Field(
        default="", description="Brief quote from the NDAA"
    )


class ChangeManifest(BaseModel):
    """Structured change manifest from an NDAA section."""
    mandated_changes: list[MandatedChange] = Field(default_factory=list)
    definitions_introduced: list[str] = Field(default_factory=list)
    authorities_amended: list[str] = Field(default_factory=list)
    effective_date: str = Field(
        default="", description="When the changes take effect"
    )


class SectionAssignment(BaseModel):
    """Assignment of changes to a specific DFARS section."""
    section_index: int
    section_name: str = Field(default="")
    role: str = Field(
        description="One of: primary, secondary, cite-only, unaffected"
    )
    change_type: str = Field(
        default="none",
        description="One of: substantive, definitional, cross-reference, none",
    )
    hosts_definitions: bool = False
    cites_definitions_from: Optional[str] = None
    assigned_changes: list[str] = Field(default_factory=list)
    delegation_notes: str = Field(default="")


class CrossReference(BaseModel):
    """A cross-reference between two DFARS sections."""
    source_section: str = Field(description="Section that references another")
    target_section: str = Field(description="Section being referenced")
    note: str = Field(default="", description="Why this cross-reference exists")


class DelegationPlan(BaseModel):
    """Assignment plan mapping changes to DFARS sections."""
    assignments: list[SectionAssignment] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)
