import os
from pymongo import MongoClient
from dotenv import load_dotenv

def getMongoClient():
    load_dotenv()
    uri = os.getenv("MONGO_CLIENT_URI")
    if not uri:
        raise ValueError("MONGO_CLIENT_URI not found in environment variables")
    client = MongoClient(uri)
    return client

def insert_docs(client: MongoClient, 
                db_name: str,
                collection: str, 
                docs: list[dict]):
    db = client[db_name]
    collection = db[collection]
    collection.insert_many(docs)