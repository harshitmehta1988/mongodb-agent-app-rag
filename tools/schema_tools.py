"""Tools for discovering database and collection schema."""
from langchain_core.tools import tool


def get_list_collections_tool(db):
    """Return a tool that lists collection names in the inferyx database."""

    @tool
    def list_collections() -> str:
        """List all collection names in the inferyx database. Call this first to know which collections exist before querying or building aggregations."""
        try:
            names = db.list_collection_names()
            return f"Collections in database 'inferyx': {', '.join(names) or 'None'}"
        except Exception as e:
            return f"Error listing collections: {e}"

    return list_collections


def get_collection_schema_tool(db):
    """Return a tool that describes a collection's schema (field names and types from sample documents)."""

    @tool
    def get_collection_schema(collection_name: str, sample_size: int = 3) -> str:
        """Get the schema of a collection by sampling documents. Use this to understand field names and types before writing find queries or aggregation pipelines. For joins, get schema of both collections to see which fields can be used for $lookup (e.g. foreign key vs local field).
        Args:
            collection_name: Exact name of the collection (e.g. 'users', 'orders').
            sample_size: Number of documents to sample for schema inference (default 3).
        """
        try:
            coll = db[collection_name]
            cursor = coll.find().limit(sample_size)
            docs = list(cursor)
            if not docs:
                return f"Collection '{collection_name}' is empty or does not exist."
            lines = [f"Collection: {collection_name}", f"Sample size: {len(docs)}", ""]
            for i, doc in enumerate(docs):
                lines.append(f"--- Document {i + 1} ---")
                for k, v in doc.items():
                    t = type(v).__name__
                    if isinstance(v, dict):
                        t += f" (keys: {list(v.keys())[:8]})"
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        t += f" (list of dicts, first keys: {list(v[0].keys())[:5]})"
                    lines.append(f"  {k}: {t}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            return f"Error getting schema for '{collection_name}': {e}"

    return get_collection_schema
