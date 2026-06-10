import json
import os
import re

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.mongo_utils import getMongoClient, get_docs_by_year, update_citations_by_docid

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:9379/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4-12b,gpu")

llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv("LLM_API_KEY", "none"))


class Citations(BaseModel):
    us_code: list[str] = Field(default_factory=list, description='U.S. Code citations, e.g. "10 U.S.C. 3201"')
    public_law: list[str] = Field(default_factory=list, description='Public Law numbers, e.g. "115-91"')
    statutes_at_large: list[str] = Field(default_factory=list, description='Statutes at Large citations, e.g. "131 Stat. 1283"')
    far: list[str] = Field(default_factory=list, description='Federal Acquisition Regulation citations, e.g. "15.209"')
    dfars: list[str] = Field(default_factory=list, description='DFARS citations, e.g. "252.203-7000"')
    ndaa: list[str] = Field(default_factory=list, description='References to other NDAA sections as "YEAR_SECTION", e.g. "2017_847"')
    other: list[str] = Field(default_factory=list, description="Any other legal citations that do not fit the above categories")


SYSTEM_PROMPT = """You are a legal citation extractor for U.S. defense legislation.
Given the text of a section of a National Defense Authorization Act (NDAA), extract every legal citation it contains, grouped by type:

- us_code: U.S. Code citations, normalized like "10 U.S.C. 3201" (title, "U.S.C.", section number).
- public_law: Public Law numbers, normalized like "115-91" (no "Public Law" prefix).
- statutes_at_large: Statutes at Large citations like "131 Stat. 1283".
- far: Federal Acquisition Regulation citations like "15.209" or "19.201".
- dfars: Defense FAR Supplement citations like "252.203-7000".
- ndaa: References to sections of other NDAAs, formatted "YEAR_SECTION" using the fiscal year, e.g. section 847 of the NDAA for Fiscal Year 2017 becomes "2017_847".
- other: any remaining legal citations (CFR, Executive Orders, other named acts, etc.) as written.

List each distinct citation once. If a category has no citations, return an empty list for it.
Respond with JSON only."""


def build_user_prompt(doc: dict) -> str:
    section = doc["section"]
    return (
        f"NDAA for Fiscal Year {doc['fiscal_year']}\n"
        f"Section {section['number']}: {section['heading']}\n\n"
        f"{section['text']}"
    )


def extract_citations_structured(doc: dict) -> Citations:
    """Extract citations using native structured outputs (json_schema)."""
    response = llm_client.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(doc)},
        ],
        response_format=Citations,
        temperature=0,
    )
    return response.choices[0].message.parsed


def extract_citations_json_fallback(doc: dict) -> Citations:
    """Fallback for servers without json_schema support: ask for JSON and parse it."""
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(doc) + "\n\nRespond with a single JSON object with keys: "
             "us_code, public_law, statutes_at_large, far, dfars, ndaa, other. Each value is a list of strings."},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
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

    for year in range(2000, 2026):
        tqdm.write(f"Extracting citations for {year}")
        docs = get_docs_by_year(mongo_client, db_name, collection_name, year)
        for doc in tqdm(docs):
            existing = doc.get("extracted_citations") or {}
            try:
                citations = extract_citations(doc)
            except Exception as e:
                tqdm.write(f"  FAILED {doc['_id']}: {e}")
                continue

            # Merge so ingestion-time fields like usc_notes are preserved
            merged = {**existing, **citations.model_dump()}
            update_citations_by_docid(mongo_client, db_name, collection_name, doc["_id"], merged)
