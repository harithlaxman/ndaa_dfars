import os

from dotenv import load_dotenv
from pymongo import MongoClient, TEXT
from typing import List, Dict, Any, Optional

def getMongoClient() -> MongoClient:
    """Initializes and returns a MongoDB client using environment variables."""
    load_dotenv()
    uri = os.getenv("MONGO_CLIENT_URI")
    if not uri:
        raise ValueError("MONGO_CLIENT_URI not found in environment variables")
    client = MongoClient(uri)
    return client

def insert_docs(client: MongoClient, 
                db_name: str,
                collection_name: str, 
                docs: List[Dict[str, Any]]) -> None:
    """Inserts a list of documents into the specified database and collection."""
    db = client[db_name]
    collection = db[collection_name]
    collection.insert_many(docs)

def get_doc_by_id(client: MongoClient, 
                  db_name: str, 
                  collection_name: str, 
                  doc_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single document by its exact _id."""
    db = client[db_name]
    collection = db[collection_name]
    
    query = {"_id": doc_id}
    return collection.find_one(query)

def get_doc_by_year_section(client: MongoClient,
                            db_name: str,
                            collection_name: str,
                            year: int,
                            section: str) -> Optional[Dict[str, Any]]:
    """Fetches a single document by its year and section."""
    return get_doc_by_id(client, db_name, collection_name, f"{year}_{section}")

def get_docs_by_year(client: MongoClient, 
                     db_name: str, 
                     collection_name: str, 
                     year: int) -> List[Dict[str, Any]]:
    """Retrieves all sections belonging to a specific fiscal year."""
    db = client[db_name]
    collection = db[collection_name]
    
    query = {"fiscal_year": year}
    return list(collection.find(query))

def get_docs_by_metadata(client: MongoClient, 
                         db_name: str, 
                         collection_name: str, 
                         metadata_key: str, 
                         metadata_value: str) -> List[Dict[str, Any]]:
    """
    Searches for documents matching a specific metadata field.
    Example: metadata_key="title", metadata_value="VIII"
    """
    db = client[db_name]
    collection = db[collection_name]
    
    # Dot notation allows MongoDB to query nested JSON objects
    query = {f"metadata.{metadata_key}": metadata_value}
    return list(collection.find(query))

def get_docs_by_citation(client: MongoClient, 
                         db_name: str, 
                         collection_name: str, 
                         citation_type: str, 
                         citation_value: str) -> List[Dict[str, Any]]:
    """
    Finds documents that cite a specific law or code.
    Example: citation_type="us_code", citation_value="10 U.S.C. 3201"
    """
    db = client[db_name]
    collection = db[collection_name]
    
    query = {f"extracted_citations.{citation_type}": citation_value}
    return list(collection.find(query))

def create_ndaa_text_index(client: MongoClient, 
                           db_name: str, 
                           collection_name: str) -> str:
    """
    Creates a text index on the section text and heading for keyword searching.
    You only need to run this once after ingesting your data.
    """
    db = client[db_name]
    collection = db[collection_name]
    
    # Creates an index on multiple fields
    index_name = collection.create_index([
        ("section.text", TEXT),
        ("section.heading", TEXT)
    ])
    return index_name

def search_docs_by_keyword(client: MongoClient, 
                           db_name: str, 
                           collection_name: str, 
                           search_phrase: str,
                           year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Performs a full-text search across the indexed fields (text and heading).
    Optionally filters the results by a specific fiscal year.
    Requires create_ndaa_text_index() to have been run previously.
    """
    db = client[db_name]
    collection = db[collection_name]
    
    # Base query: Wrapping the phrase in escaped quotes forces an exact phrase match
    query: Dict[str, Any] = {"$text": {"$search": f'"{search_phrase}"'}}
    
    # If a year was passed in, add it to the query dictionary
    if year is not None:
        query["fiscal_year"] = year
    
    # We return the list of matching documents
    return list(collection.find(query))


def update_citations_by_docid(client: MongoClient, 
                           db_name: str, 
                           collection_name: str, 
                           doc_id: str, 
                           new_citations: Dict[str, Any]) -> bool:
    """
    Updates ONLY the 'extracted_citations' field of a specific document by its ID.
    Returns True if the document was successfully modified.
    """
    db = client[db_name]
    collection = db[collection_name]
    
    # 1. Target the specific document
    filter_query = {"_id": doc_id}
    
    # 2. Use $set to ensure ONLY the citations field is overwritten
    update_operation = {"$set": {"extracted_citations": new_citations}}
    
    # 3. Execute the update
    result = collection.update_one(filter_query, update_operation)
    
    # Returns True if a document was found AND changed
    return result.modified_count > 0