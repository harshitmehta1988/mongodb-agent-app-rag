"""Build the schema/metadata vector index for RAG."""
from config import get_database, SCHEMA_RAG_COLLECTION
from .embeddings import get_embeddings


def _infer_schema_text(db, collection_name: str, sample_size: int = 3) -> str:
    """Produce a single searchable text blob for a collection (name + field names and types)."""
    coll = db[collection_name]
    cursor = coll.find().limit(sample_size)
    docs = list(cursor)
    if not docs:
        return f"Collection {collection_name} (empty)"
    parts = [f"Collection: {collection_name}. Fields:"]
    seen = set()
    for doc in docs:
        for k, v in doc.items():
            if k in seen:
                continue
            seen.add(k)
            t = type(v).__name__
            if isinstance(v, dict):
                t += " (keys: " + ", ".join(list(v.keys())[:6]) + ")"
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                t += " (list of objects)"
            parts.append(f"  {k}: {t}")
    return "\n".join(parts)


def build_schema_index(sample_size: int = 3) -> int:
    """
    For each collection in the DB, build a schema text, embed it with Voyage, and upsert into
    schema_metadata with an 'embedding' field. Creates one document per collection.
    Returns the number of documents written. Requires Atlas Vector Search index on schema_metadata.
    """
    db = get_database()
    coll_names = db.list_collection_names()
    # Skip RAG and system collections
    skip = {SCHEMA_RAG_COLLECTION, "query_examples", "system.indexes"}
    to_index = [c for c in coll_names if c not in skip and not c.startswith("system.")]
    if not to_index:
        return 0
    texts = [_infer_schema_text(db, c, sample_size) for c in to_index]
    embeddings = get_embeddings(texts)
    collection = db[SCHEMA_RAG_COLLECTION]
    written = 0
    for name, text, emb in zip(to_index, texts, embeddings):
        if not emb:
            continue
        doc = {
            "collection_name": name,
            "text": text,
            "embedding": emb,
        }
        collection.update_one(
            {"collection_name": name},
            {"$set": doc},
            upsert=True,
        )
        written += 1
    return written
