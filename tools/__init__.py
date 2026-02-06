from .schema_tools import get_list_collections_tool, get_collection_schema_tool
from .query_tools import get_execute_find_tool, get_execute_aggregation_tool


def get_all_tools(db):
    """Return all agent tools bound to the given database."""
    return [
        get_list_collections_tool(db),
        get_collection_schema_tool(db),
        get_execute_find_tool(db),
        get_execute_aggregation_tool(db),
    ]
