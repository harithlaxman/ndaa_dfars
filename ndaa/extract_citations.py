import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.mongo_utils import getMongoClient, get_docs_by_year, update_citations_by_docid
from utils.openai import connect_to_openai, get_structured_response, get_response

llm_client = connect_to_openai()

CitationType = Literal[
    "us_code", "public_law", "statutes_at_large", "far", "dfars", "ndaa", "other"
]


class AlternateCitation(BaseModel):
    value: str = Field(description="The alternate name for the same citation, normalized for its type")
    type: CitationType = Field(description="Citation type of the alternate name")


class Citation(BaseModel):
    value: str = Field(description="The normalized primary citation")
    type: CitationType = Field(description="Citation type of the primary value")
    alternate: Optional[AlternateCitation] = Field(
        default=None,
        description="An alternate name referring to this same citation, if one exists",
    )


class Citations(BaseModel):
    citations: list[Citation] = Field(default_factory=list)


SYSTEM_PROMPT = """You are a legal citation extractor for U.S. defense legislation.
Given the text of a section of a National Defense Authorization Act (NDAA), extract every legal citation it contains as a flat list of citation objects.

Each citation object has:
- value: the normalized citation text.
- type: one of the citation types below.
- alternate (optional): if the same real-world law/citation is named more than one way, record the citation ONCE and put the other name here as {value, type}. Otherwise omit it (null).

Citation types and how to normalize each value:
- us_code: U.S. Code citations, normalized like "10 U.S.C. 3201" (title, "U.S.C.", section number).
- public_law: Public Law numbers, normalized like "115-91" (no "Public Law" prefix).
- statutes_at_large: Statutes at Large citations like "131 Stat. 1283".
- far: Federal Acquisition Regulation citations like "15.209" or "19.201".
- dfars: Defense FAR Supplement citations like "252.203-7000".
- ndaa: References to sections of other NDAAs, formatted "YEAR_SECTION" using the fiscal year, e.g. section 847 of the NDAA for Fiscal Year 2017 becomes "2017_847".
- other: any remaining legal citations (CFR, Executive Orders, other named acts, etc.) as written.

Key rule — do not duplicate the same citation under two names. In particular, an NDAA is the same thing as a Public Law:
- When an NDAA is cited together with its Public Law number (commonly given in brackets, e.g. "the National Defense Authorization Act for Fiscal Year 2017 (Public Law 114-328)"), store the Public Law as the primary value (type "public_law") and the NDAA as the alternate (type "ndaa", "YEAR_SECTION" format). When the citation is to a specific NDAA section, use that section in the alternate (e.g. "2017_847").
- When only the NDAA is named (no Public Law given), store it as the primary value with type "ndaa" and no alternate.
- Apply the same single-record-with-alternate approach to any other citation that appears under more than one name.

List each distinct citation once. If there are no citations, return an empty list.
Respond with JSON only."""


def build_user_prompt(doc: dict) -> str:
    section = doc["section"]
    return (
        f"NDAA Fiscal Year: {doc['fiscal_year']}\n"
        f"Section Number: {section['number']}\n"
        f"Section Title: {section['heading']}\n\n"
        f"Section Text:\n{section['text']}"
    )


def extract_citations_structured(doc: dict) -> Citations:
    """Extract citations using Azure OpenAI structured outputs (responses.parse)."""
    return get_structured_response(llm_client, SYSTEM_PROMPT, build_user_prompt(doc), Citations)


def extract_citations_json_fallback(doc: dict) -> Citations:
    """Fallback: ask for JSON text and parse it manually."""
    user_content = (
        build_user_prompt(doc) + "\n\nRespond with a single JSON object with one key, "
        '"citations", whose value is a list of objects. Each object has "value" (string), '
        '"type" (one of us_code, public_law, statutes_at_large, far, dfars, ndaa, other), '
        'and optionally "alternate" ({"value": string, "type": <same type set>}).'
    )
    content = get_response(llm_client, SYSTEM_PROMPT, user_content)
    # Strip markdown code fences and any surrounding prose
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {content[:200]}")
    return Citations.model_validate(json.loads(match.group(0)))


def extract_citations(doc: dict) -> Citations:
    try:
        return extract_citations_structured(doc)
    except Exception:
        return extract_citations_json_fallback(doc)


if __name__ == "__main__":
    mongo_client = getMongoClient()
    db_name = "ndaa_dfars"
    collection_name = "ndaas"

    for year in range(2010, 2026):
        tqdm.write(f"Extracting citations for {year}")
        docs = get_docs_by_year(mongo_client, db_name, collection_name, year)
        for doc in tqdm(docs):
            try:
                citations = extract_citations(doc)
            except Exception as e:
                tqdm.write(f"  FAILED {doc['_id']}: {e}")
                continue

            # Overwrite extracted_citations wholesale with the citation list;
            # usc_notes is re-derivable from ingestion and not needed here.
            update_citations_by_docid(
                mongo_client, db_name, collection_name, doc["_id"], citations.model_dump()["citations"]
            )

