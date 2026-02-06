# MongoDB Agent with RAG (Vector Search)

This repo extends the [mongodb-agent-app](https://github.com/your-org/mongodb-agent-app) agent with **Schema / metadata RAG** and **Query-example / documentation RAG** using **Voyage AI** embeddings and **MongoDB Atlas Vector Search**. The UI matches the original app; only the backend adds RAG context to improve accuracy and reduce tool calls.

## Features

- **Schema RAG**: Embeddings of collection/field metadata; for each user question the agent retrieves the most relevant schema chunks and injects them into the system prompt so the model can prioritize the right collections and fields.
- **Query-example RAG**: Embeddings of (natural language question, query/tool usage) pairs; similar past examples are retrieved and injected as few-shot context to guide query shape and tool usage.
- **Voyage AI**: All embeddings use the Voyage AI model (`voyage-3`, 1024 dimensions).
- **Atlas Vector Search**: Vectors are stored in MongoDB; retrieval uses the `$vectorSearch` aggregation stage.

## Setup

### 1. Environment

```bash
cp .env.example .env
# Edit .env: MONGODB_URI, ANTHROPIC_API_KEY, VOYAGE_API_KEY
```

### 2. Atlas Vector Search indexes

Create two vector search indexes in MongoDB Atlas on the **inferyx** database. In Atlas: go to your cluster → **Search** tab (or **Database** → **Browse Collections** → select collection → **Create Search Index**) → **JSON Editor**, then use the definitions below.

| Collection        | Index name                      | Vector path | Dimensions |
|-------------------|----------------------------------|-------------|------------|
| `schema_metadata` | `schema_metadata_vector_index`  | `embedding` | 1024       |
| `query_examples`  | `query_examples_vector_index`   | `embedding` | 1024       |

**Index 1 – Schema metadata**

- **Database:** `inferyx` · **Collection:** `schema_metadata` · **Index name:** `schema_metadata_vector_index`

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024 }
  ]
}
```

**Index 2 – Query examples**

- **Database:** `inferyx` · **Collection:** `query_examples` · **Index name:** `query_examples_vector_index`

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024 }
  ]
}
```

The `schema_metadata` and `query_examples` collections are created when you first run the RAG build or ingest script (with `VOYAGE_API_KEY` set). Create the two indexes above **after** those collections exist, or create the collections manually first; the indexes can be created as soon as the collection exists (even if empty).

### 3. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 4. Load data and build RAG indexes

**Option A – Ingest sample data (recommended for first run)**  
Loads the four collections from `sampledata/` into the **inferyx** database, then builds the RAG indexes so vector retrieval works for those collections.

```bash
python3 scripts/ingest_sampledata.py
```

- **Collections created:** `datapod`, `datasource`, `dataset`, `vizpods` (from `sampledata/datapod.json`, `datasource.json`, `dataset_10.json`, `vizpods_10.json`).
- **RAG:** Populates `schema_metadata` and `query_examples` with vector embeddings (requires `VOYAGE_API_KEY`). If `VOYAGE_API_KEY` is missing, the collections are ingested but RAG indexes stay empty; set the key and run again or run `python3 scripts/build_rag_indexes.py`.
- **Flags:** `--no-rag` to skip RAG build (ingest only). `--no-replace` to append to collections instead of replacing.

**Option B – Build RAG indexes only**  
Use when the database already has collections and you only need to refresh schema/query-example vectors.

```bash
python3 scripts/build_rag_indexes.py
```

### 5. Run the app

```bash
streamlit run app.py
```

## Project layout

- `app.py` – Streamlit UI (same as original).
- `agent.py` – LangGraph agent; before each turn it retrieves schema + query-example context and injects it into the system prompt.
- `config.py` – MongoDB, Anthropic, Voyage, and RAG collection/index names.
- `sampledata/` – Sample JSON for the four Inferyx collections (`datapod.json`, `datasource.json`, `dataset_10.json`, `vizpods_10.json`).
- `rag/` – RAG implementation:
  - `embeddings.py` – Voyage AI embed calls.
  - `retrieval.py` – Atlas `$vectorSearch` for schema and query examples.
  - `schema_index.py` – Builds `schema_metadata` from current DB collections.
  - `query_examples_index.py` – Builds `query_examples` from built-in (and optional file) examples.
- `tools/` – Same as original (list_collections, get_collection_schema, execute_find, execute_aggregation).
- `scripts/build_rag_indexes.py` – Rebuild both RAG indexes (schema + query examples).
- `scripts/ingest_sampledata.py` – Ingest sample data from `sampledata/` into the four collections, then build RAG indexes for effective vector retrieval.

## Adding your own query examples

1. Create a JSON file, e.g. `data/query_examples.json`, with a list of objects:
   - `natural_language` or `question`: the user question.
   - `query` or `pipeline` or `example_query`: the tool call or query description (string or object).

2. Rebuild the query-examples index, passing the file:

   In code:

   ```python
   from rag import build_query_examples_index
   build_query_examples_index(examples_file="data/query_examples.json")
   ```

   Or extend `scripts/build_rag_indexes.py` to pass `examples_file` into `build_query_examples_index()`.

3. Re-run the app; similar user questions will retrieve your examples.

## Optimizing response time

End-to-end latency is dominated by: **Voyage embedding** (user query → vector), **Atlas vector search**, **LLM call(s)**, and **tool execution**. Ways to improve:

| Area | What to do |
|------|------------|
| **RAG** | Query-example RAG is disabled in the agent (schema RAG only). Reduce `top_k` in `agent.py` for `retrieve_schema_context(user_text, top_k=8)` (e.g. `top_k=4`) to shrink prompt and retrieval time. |
| **Embedding** | One Voyage call per turn. Use a smaller/faster embedding model in `config.py` if supported, or cache embeddings for repeated queries (not implemented). |
| **Vector search** | Keep `numCandidates` in `rag/retrieval.py` modest; already `max(limit * 20, 100)`. Ensure the Atlas cluster and index are in the same region as the app. |
| **LLM** | In `agent.py`, `ChatAnthropic` uses `claude-sonnet-4-20250514`. Switching to a smaller/faster model (e.g. Haiku or a smaller Sonnet variant) reduces time per turn; use `max_tokens` to cap output length. |
| **Tools** | Each tool call adds a round-trip. Schema RAG helps the model pick the right collections and query shape, which can reduce the number of tool calls. |
| **Streaming** | The UI waits for the full run; the agent uses `stream(..., stream_mode="values")` and only the final state is used. Streaming tokens to the UI would not reduce total time but would improve perceived latency. |

## Notes

- The **original repo is unchanged**; this is a separate folder/repo that reuses the same agent design and UI and adds RAG on top.
- If `VOYAGE_API_KEY` is missing, RAG retrieval returns no context and the agent behaves like the original (no vector search).
- Re-run `build_rag_indexes.py` or `ingest_sampledata.py` (without `--no-rag`) after adding collections or changing schema/example data so the vector indexes stay in sync.
