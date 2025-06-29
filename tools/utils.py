from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from typing import Any, Dict, List, Union


def json_to_tools_schema(tool_json: Union[Dict[str, Any], List[Dict[str, Any]]]) -> ToolsSchema:
    """
    Convert a tool JSON (single object or list) to a ToolsSchema instance.
    """
    def convert_single_tool(tool: Dict[str, Any]) -> FunctionSchema:
        name = tool["name"]
        description = tool.get("description", "")
        api_schema = tool.get("api_schema", {})
        properties = {}
        required = []

        # Query params
        for param in api_schema.get("query_params_schema", []):
            properties[param["id"]] = {
                "type": param["type"],
                "description": param.get("description", "")
            }
            if param.get("required", False):
                required.append(param["id"])

        # Request body properties
        body_schema = api_schema.get("request_body_schema", {})
        for prop in body_schema.get("properties", []):
            properties[prop["id"]] = {
                "type": prop["type"],
                "description": prop.get("description", "")
            }
            if prop.get("required", False):
                required.append(prop["id"])
        for param in api_schema.get("path_params_schema", []):
            properties[param["id"]] = {
                "type": param["type"],
                "description": param.get("description", "")
            }
            if param.get("required", False):
                required.append(param["id"])
                
        return FunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required
        )

    if isinstance(tool_json, dict):
        tool_json = [tool_json]
    function_schemas = [convert_single_tool(tool) for tool in tool_json]
    return ToolsSchema(standard_tools=function_schemas)
