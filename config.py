"""Configuration for MongoDB, LLM, Voyage AI, and RAG collections."""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
DB_NAME = "inferyx"

# RAG: collections and vector index names (must match Atlas Vector Search index definitions)
SCHEMA_RAG_COLLECTION = "schema_metadata"
SCHEMA_RAG_INDEX_NAME = "schema_metadata_vector_index"
QUERY_EXAMPLES_COLLECTION = "query_examples"
QUERY_EXAMPLES_INDEX_NAME = "query_examples_vector_index"

# Voyage embedding model (voyage-3 = 1024 dimensions)
VOYAGE_EMBED_MODEL = "voyage-3"
VECTOR_DIMENSION = 1024


def get_mongo_client() -> MongoClient:
    """Return a MongoDB client."""
    return MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)


def get_database():
    """Return the inferyx database."""
    client = get_mongo_client()
    return client[DB_NAME]
