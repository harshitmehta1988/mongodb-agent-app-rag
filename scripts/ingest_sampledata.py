#!/usr/bin/env python3
"""
Ingest sample data from sampledata/ into the inferyx database (4 collections),
then build RAG indexes so the app's vector retrieval works effectively.

Collections: datapod, datasource, dataset, vizpods.
After ingest, builds:
  - schema_metadata: vector embeddings of collection schema (for schema RAG).
  - query_examples: vector embeddings of (question, query) pairs (for query-example RAG).

Requires: MONGODB_URI, VOYAGE_API_KEY (for embeddings). Run from repo root or with
  PYTHONPATH including repo root.
"""
from __future__ import annotations

import json
import os
import sys

# Repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bson import json_util as bson_json_util

from config import get_database, DB_NAME
from rag import build_schema_index, build_query_examples_index
from rag.query_examples_index import load_default_examples

# Path to sampledata directory (repo-relative)
SAMPLEDATA_DIR = os.path.join(_REPO_ROOT, "sampledata")

# File -> collection name
FILE_TO_COLLECTION = {
    "datapod.json": "datapod",
    "datasource.json": "datasource",
    "dataset_10.json": "dataset",
    "vizpods_10.json": "vizpods",
}

# Extra query examples for Inferyx collections (improves vector retrieval for this app)
INFERYX_QUERY_EXAMPLES = [
    {"natural_language": "List all collections in inferyx", "query": "list_collections", "tool": "list_collections"},
    {"natural_language": "What collections exist in the database?", "query": "list_collections", "tool": "list_collections"},
    {"natural_language": "Show schema of datapod collection", "query": "get_collection_schema(collection_name='datapod')", "tool": "get_collection_schema"},
    {"natural_language": "Show schema of datasource collection", "query": "get_collection_schema(collection_name='datasource')", "tool": "get_collection_schema"},
    {"natural_language": "Show schema of dataset collection", "query": "get_collection_schema(collection_name='dataset')", "tool": "get_collection_schema"},
    {"natural_language": "Show schema of vizpods collection", "query": "get_collection_schema(collection_name='vizpods')", "tool": "get_collection_schema"},
    {"natural_language": "List all documents in datasource", "query": 'execute_find(collection_name="datasource", filter_json="{}")', "tool": "execute_find"},
    {"natural_language": "List all documents in dataset", "query": 'execute_find(collection_name="dataset", filter_json="{}")', "tool": "execute_find"},
    {"natural_language": "Find active datasources", "query": 'execute_find(collection_name="datasource", filter_json=\'{"active": "Y"}\')', "tool": "execute_find"},
    {"natural_language": "Count documents in datasource", "query": 'execute_aggregation(collection_name="datasource", pipeline_json=\'[{"$count": "total"}]\')', "tool": "execute_aggregation"},
    {"natural_language": "Count documents in dataset", "query": 'execute_aggregation(collection_name="dataset", pipeline_json=\'[{"$count": "total"}]\')', "tool": "execute_aggregation"},
]


def _parse_extended_json(s: str):
    """Parse MongoDB extended JSON string (supports $oid, $date) into a Python object."""
    return bson_json_util.loads(s)


def _load_json_documents(path: str) -> list[dict]:
    """
    Load one or more JSON documents from a file. Handles:
    - Single JSON object
    - JSON array of objects
    - Concatenated JSON objects (one per block, separated by blank lines)
    Uses MongoDB extended JSON for _id.$oid and $date.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.strip()
    if not raw:
        return []

    # Try single object or array first
    try:
        obj = _parse_extended_json(raw)
        if isinstance(obj, list):
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass

    # Multiple top-level objects: use JSONDecoder.raw_decode to find boundaries (avoids splitting on nested braces)
    docs = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw):
        pos = len(raw) - len(raw[pos:].lstrip())
        if pos >= len(raw):
            break
        try:
            _, end = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse at position {pos} in {path}: {e}") from e
        chunk = raw[pos:end]
        docs.append(_parse_extended_json(chunk))
        pos = end
    return docs


def ingest_collection(db, collection_name: str, filepath: str, replace: bool = True) -> int:
    """
    Load JSON from filepath and insert into db[collection_name].
    If replace=True, drops existing collection and inserts fresh. Returns count inserted.
    """
    coll = db[collection_name]
    if replace:
        coll.delete_many({})
    docs = _load_json_documents(filepath)
    if not docs:
        return 0
    coll.insert_many(docs)
    return len(docs)


def main(replace: bool = True, build_rag: bool = True) -> None:
    if not os.path.isdir(SAMPLEDATA_DIR):
        print(f"Sampledata directory not found: {SAMPLEDATA_DIR}")
        sys.exit(1)

    db = get_database()
    total = 0
    for filename, coll_name in FILE_TO_COLLECTION.items():
        path = os.path.join(SAMPLEDATA_DIR, filename)
        if not os.path.isfile(path):
            print(f"Skip (file not found): {filename}")
            continue
        n = ingest_collection(db, coll_name, path, replace=replace)
        total += n
        print(f"  {filename} -> {coll_name}: {n} document(s)")

    print(f"Ingested {total} documents across {len(FILE_TO_COLLECTION)} collections.")

    if build_rag:
        print("Building RAG indexes for effective vector retrieval...")
        n_schema = build_schema_index(sample_size=5)
        print(f"  schema_metadata: {n_schema} collection(s) indexed with vector embeddings.")
        if n_schema == 0:
            print("  → schema_metadata was not created. Set VOYAGE_API_KEY in .env and re-run to create it.")
        # Default examples + Inferyx-specific examples for datapod/datasource/dataset/vizpods
        all_examples = load_default_examples() + INFERYX_QUERY_EXAMPLES
        n_examples = build_query_examples_index(examples=all_examples)
        print(f"  query_examples: {n_examples} example(s) indexed with vector embeddings.")
        if n_examples == 0:
            print("  → query_examples was not created. Set VOYAGE_API_KEY in .env and re-run to create it.")
        print("Done. Schema RAG and query-example RAG are ready for the app.")
    else:
        print("Skipping RAG build. Run: python scripts/build_rag_indexes.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Ingest sampledata into inferyx and build RAG indexes.")
    p.add_argument("--no-replace", action="store_true", help="Append instead of replacing collection contents")
    p.add_argument("--no-rag", action="store_true", help="Do not build schema/query-example vector indexes after ingest")
    args = p.parse_args()
    main(replace=not args.no_replace, build_rag=not args.no_rag)
