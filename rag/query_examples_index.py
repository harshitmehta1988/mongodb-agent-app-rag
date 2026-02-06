"""Build the query-examples vector index for RAG (few-shot / documentation RAG)."""
import json
import os
from config import get_database, QUERY_EXAMPLES_COLLECTION
from .embeddings import get_embeddings


def _example_to_text(ex: dict) -> str:
    """Single searchable text for an example (natural language + query)."""
    nl = ex.get("natural_language") or ex.get("question") or ""
    q = ex.get("query") or ex.get("pipeline") or ex.get("example_query") or ""
    if isinstance(q, (list, dict)):
        q = json.dumps(q)[:500]
    return f"Question: {nl}. Query: {q}"


def load_default_examples() -> list[dict]:
    """Return built-in example (question, query) pairs for common patterns."""
    return [
        {
            "natural_language": "List all collections in the database",
            "query": "list_collections",
            "tool": "list_collections",
        },
        {
            "natural_language": "What collections exist?",
            "query": "list_collections",
            "tool": "list_collections",
        },
        {
            "natural_language": "Show me the schema of the users collection",
            "query": "get_collection_schema(collection_name='users')",
            "tool": "get_collection_schema",
        },
        {
            "natural_language": "Find all documents in users where status is active",
            "query": 'execute_find(collection_name="users", filter_json=\'{"status": "active"}\')',
            "tool": "execute_find",
        },
        {
            "natural_language": "List all users",
            "query": 'execute_find(collection_name="users", filter_json="{}")',
            "tool": "execute_find",
        },
        {
            "natural_language": "Join orders with users and show user name and order total",
            "query": (
                'execute_aggregation(collection_name="orders", pipeline_json=\''
                '[{"$lookup": {"from": "users", "localField": "userId", "foreignField": "_id", "as": "user"}}, '
                '{"$unwind": "$user"}, {"$project": {"user.name": 1, "total": 1, "_id": 0}}]\''
            ),
            "tool": "execute_aggregation",
        },
        {
            "natural_language": "How many orders per customer?",
            "query": (
                'execute_aggregation(collection_name="orders", pipeline_json=\''
                '[{"$group": {"_id": "$userId", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]\''
            ),
            "tool": "execute_aggregation",
        },
        {
            "natural_language": "Count documents in users collection",
            "query": (
                'execute_aggregation(collection_name="users", pipeline_json=\''
                '[{"$count": "total"}]\''
            ),
            "tool": "execute_aggregation",
        },
    ]


def build_query_examples_index(examples: list[dict] | None = None, examples_file: str | None = None) -> int:
    """
    Embed each example (natural_language + query) and upsert into query_examples with 'embedding'.
    examples: list of dicts with keys e.g. natural_language, query (or question, pipeline).
    examples_file: optional path to JSON file with list of such dicts (merged with defaults if examples not provided).
    Returns number of documents written.
    """
    if examples is None:
        examples = load_default_examples()
    if examples_file and os.path.isfile(examples_file):
        with open(examples_file) as f:
            file_examples = json.load(f)
        if isinstance(file_examples, list):
            examples = examples + file_examples
    if not examples:
        return 0
    texts = [_example_to_text(ex) for ex in examples]
    embeddings = get_embeddings(texts)
    db = get_database()
    collection = db[QUERY_EXAMPLES_COLLECTION]
    written = 0
    for i, (ex, emb) in enumerate(zip(examples, embeddings)):
        if not emb:
            continue
        nl = ex.get("natural_language") or ex.get("question") or ""
        q = ex.get("query") or ex.get("example_query") or ""
        sid = f"{nl[:80]}_{str(q)[:50]}"
        doc = {
            "_rag_id": sid,
            **{k: v for k, v in ex.items() if k != "embedding"},
            "embedding": emb,
        }
        collection.update_one(
            {"_rag_id": sid},
            {"$set": doc},
            upsert=True,
        )
        written += 1
    return written
