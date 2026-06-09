import requests

import pandas as pd
from tqdm import tqdm

from utils.ndaa_html_parser import parse_plaw_html
from utils.mongo_utils import getMongoClient, insert_docs

PLAW_URL = "https://www.govinfo.gov/link/plaw/"
PLAW_TYPE = "/public/"
PLAW_LINK_TYPE = "?link-type=html"

def get_ndaa_htmls(plaw_df: pd.DataFrame):
    ndaa_htmls = {}
    for i, row in tqdm(plaw_df.iterrows(), total=len(plaw_df), desc="Fetching NDAA HTMLs"):
        year = row["Fiscal year"]
        plaw = row["Public Law"].strip("Pub. L. ")
        congresnum = plaw[:-3].strip("-")
        lawnum = plaw[4:]
        
        req_url = PLAW_URL + congresnum + PLAW_TYPE + lawnum + PLAW_LINK_TYPE
        
        response = requests.get(req_url)
        if (response.status_code == 200):
            ndaa_htmls[year] = {"html": response.text, "plaw": plaw}
        else:
            print("couldn't fetch for year: ", year)
    
    return ndaa_htmls

def format_htmls(ndaa_htmls):
    formatted_documents = []
    ids = set()
    for year, data in tqdm(ndaa_htmls.items(), total=len(ndaa_htmls), desc="Formatting NDAA HTMLs"):
        plaw = data["plaw"]
        parsed = parse_plaw_html(data["html"], year)
        for item in parsed:
            section_num = item["section"]
            id = f"{year}_{section_num}"
            if id in ids:
                id += "_"
            ids.add(id)
            doc = {
                "_id": id,
                "doc_type": "NDAA",
                "fiscal_year": year,
                "metadata": {
                    "plaw": plaw,
                    "ndaa_title": item["title"],
                    "ndaa_subtitle": item["subtitle"]
                },
                "section": {
                    "number": section_num,
                    "heading": item["heading"].strip(),
                    "text": item["text"].strip()
                },
                "extracted_citations": {
                    "usc_notes": item["usc_notes"]
                }
            }
            formatted_documents.append(doc)
    return formatted_documents

if __name__ == "__main__":
    client = getMongoClient()
    db_name = "ndaa_dfars"
    collection_name = "ndaas"

    plaw_df = pd.read_csv("./ndaa_plaws.csv")
    
    ndaa_htmls = get_ndaa_htmls(plaw_df)
    formatted_documents = format_htmls(ndaa_htmls)

    insert_docs(client, db_name, collection_name, formatted_documents)