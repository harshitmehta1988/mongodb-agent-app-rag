"""Tools for executing MongoDB find and aggregation (including $lookup joins)."""
import json
from bson import ObjectId
from langchain_core.tools import tool


def _serialize(obj):
    """Convert MongoDB types to JSON-serializable (e.g. ObjectId -> str)."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def get_execute_find_tool(db):
    """Return a tool that runs a find query on a collection."""

    @tool
    def execute_find(collection_name: str, filter_json: str, projection_json: str = "{}", limit: int = 50) -> str:
        """Execute a MongoDB find query on a single collection.
        Args:
            collection_name: Name of the collection.
            filter_json: JSON object for the query filter (e.g. '{"status": "active"}' or '{}' for all).
            projection_json: Optional JSON object for projection (e.g. '{"name": 1, "email": 1, "_id": 0}').
            limit: Maximum number of documents to return (default 50).
        """
        try:
            filt = json.loads(filter_json) if filter_json.strip() else {}
            proj = json.loads(projection_json) if projection_json.strip() else {}
            coll = db[collection_name]
            cursor = coll.find(filt, proj if proj else None).limit(limit)
            docs = list(cursor)
            serialized = _serialize(docs)
            return json.dumps(serialized, indent=2, default=str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON in filter or projection: {e}"
        except Exception as e:
            return f"Error executing find: {e}"

    return execute_find


def get_execute_aggregation_tool(db):
    """Return a tool that runs an aggregation pipeline (supports $lookup for joins)."""

    @tool
    def execute_aggregation(collection_name: str, pipeline_json: str, limit_results: int = 100) -> str:
        """Execute a MongoDB aggregation pipeline on a collection. Use this for aggregations, grouping, and JOINs between two collections via $lookup.
        For a join: use $lookup with from (other collection), localField, foreignField, and as (array name for joined docs).
        Args:
            collection_name: Name of the primary collection to run the pipeline on.
            pipeline_json: JSON array of aggregation stages (e.g. '[{"$match": {"status": "active"}}, {"$lookup": {"from": "users", "localField": "userId", "foreignField": "_id", "as": "user"}}, {"$limit": 20}]').
            limit_results: Optional cap on result size; add a $limit stage if your pipeline does not include one (default 100).
        """
        try:
            pipeline = json.loads(pipeline_json)
            if not isinstance(pipeline, list):
                return "pipeline_json must be a JSON array of stages."
            has_limit = any(s.get("$limit") is not None for s in pipeline)
            if not has_limit and limit_results:
                pipeline = pipeline + [{"$limit": limit_results}]
            coll = db[collection_name]
            cursor = coll.aggregate(pipeline)
            docs = list(cursor)
            serialized = _serialize(docs)
            return json.dumps(serialized, indent=2, default=str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON pipeline: {e}"
        except Exception as e:
            return f"Error executing aggregation: {e}"

    return execute_aggregation
