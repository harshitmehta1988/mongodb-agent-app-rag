#!/usr/bin/env python3
"""Build schema and query-example RAG indexes (run once or after schema/example changes)."""
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import build_schema_index, build_query_examples_index


def main():
    print("Building schema metadata index...")
    n_schema = build_schema_index(sample_size=3)
    print(f"  Written {n_schema} schema documents.")

    print("Building query examples index...")
    n_examples = build_query_examples_index()
    print(f"  Written {n_examples} query example documents.")

    print("Done. Ensure Atlas Vector Search indexes exist (see README).")


if __name__ == "__main__":
    main()
