"""Atlas Vector Search retrieval for schema and query-example RAG."""
from config import (
    get_database,
    SCHEMA_RAG_COLLECTION,
    SCHEMA_RAG_INDEX_NAME,
    QUERY_EXAMPLES_COLLECTION,
    QUERY_EXAMPLES_INDEX_NAME,
)
from .embeddings import get_embedding


def _vector_search(db, collection_name: str, index_name: str, query_embedding: list[float], limit: int = 5):
    """Run Atlas $vectorSearch on the given collection. Returns list of docs (with score if present)."""
    if not query_embedding:
        return []
    coll = db[collection_name]
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(limit * 20, 100),
                "limit": limit,
            }
        },
        {"$project": {"embedding": 0}},  # drop vector to reduce payload
    ]
    try:
        return list(coll.aggregate(pipeline))
    except Exception:
        return []


def retrieve_schema_context(user_query: str, top_k: int = 8) -> str:
    """
    Retrieve relevant schema/metadata chunks for the user query via Atlas Vector Search.
    Returns a formatted string to inject into the agent system prompt.
    """
    db = get_database()
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return ""
    docs = _vector_search(db, SCHEMA_RAG_COLLECTION, SCHEMA_RAG_INDEX_NAME, query_embedding, limit=top_k)
    if not docs:
        return ""
    lines = [
        "Relevant schema metadata (from vector search; use this to prioritize which collections/fields to use):",
        "",
    ]
    for d in docs:
        text = d.get("text") or d.get("description") or ""
        collection = d.get("collection_name") or d.get("collection") or ""
        if text:
            lines.append(f"- [{collection}] {text}")
    return "\n".join(lines) if len(lines) > 2 else ""


def retrieve_query_examples_context(user_query: str, top_k: int = 3) -> str:
    """
    Retrieve similar past query examples (natural language + query) via Atlas Vector Search.
    Returns a formatted string to inject as few-shot context.
    """
    db = get_database()
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return ""
    docs = _vector_search(db, QUERY_EXAMPLES_COLLECTION, QUERY_EXAMPLES_INDEX_NAME, query_embedding, limit=top_k)
    if not docs:
        return ""
    lines = [
        "Similar example questions and how they were answered (use as reference for tool usage and query shape):",
        "",
    ]
    for d in docs:
        nl = d.get("natural_language") or d.get("question") or ""
        q = d.get("query") or d.get("pipeline") or d.get("example_query") or ""
        if nl or q:
            lines.append(f"Q: {nl}")
            if q:
                lines.append(f"  → {q}")
            lines.append("")
    return "\n".join(lines).strip() if len(lines) > 2 else ""
