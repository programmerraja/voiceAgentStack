import aiohttp
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

class Tool(ABC):
    """
    Abstract base class for all tools.
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @classmethod
    @abstractmethod
    def from_json(cls, tool_json: Dict[str, Any]) -> 'Tool':
        pass

    @abstractmethod
    def to_schema(self) -> FunctionSchema:
        pass

    @abstractmethod
    async def execute(self, params: dict = None, body: dict = None, headers: dict = None, timeout: float = None) -> Any:
        pass

class WebhookTool(Tool):
    def __init__(self, name: str, description: str, api_schema: dict, response_timeout_secs: Optional[float] = None):
        super().__init__(name, description)
        self.api_schema = api_schema
        self.response_timeout_secs = response_timeout_secs or 20

    @classmethod
    def from_json(cls, tool_json: Dict[str, Any]) -> 'WebhookTool':
        name = tool_json["name"]
        description = tool_json.get("description", "")
        api_schema = tool_json.get("api_schema", {})
        response_timeout_secs = tool_json.get("response_timeout_secs", 20)
        return cls(name, description, api_schema, response_timeout_secs)

    def to_schema(self) -> FunctionSchema:
        properties = {}
        required = []
        api_schema = self.api_schema

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

        # Path params
        for param in api_schema.get("path_params_schema", []):
            properties[param["id"]] = {
                "type": param["type"],
                "description": param.get("description", "")
            }
            if param.get("required", False):
                required.append(param["id"])

        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=properties,
            required=required
        )

    async def execute(self, params: dict = None, body: dict = None, headers: dict = None, timeout: float = None) -> Any:
        api_schema = self.api_schema
        url = api_schema.get("url")
        method = api_schema.get("method", "GET").upper()
        path_params = api_schema.get("path_params_schema", [])
        query_params = params or {}
        request_body = body or {}
        request_headers = headers or {}

        # Substitute path params if any
        path = api_schema.get("path", "")
        if path_params:
            for param in path_params:
                param_id = param["id"]
                if param_id in query_params:
                    path = path.replace(f"{{{param_id}}}", str(query_params[param_id]))
        full_url = url + path

        req_timeout = timeout or self.response_timeout_secs

        async with aiohttp.ClientSession() as session:
            try:
                if method == "GET":
                    async with session.get(full_url, params=query_params, headers=request_headers, timeout=req_timeout) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                elif method == "POST":
                    async with session.post(full_url, params=query_params, json=request_body, headers=request_headers, timeout=req_timeout) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                elif method == "PUT":
                    async with session.put(full_url, params=query_params, json=request_body, headers=request_headers, timeout=req_timeout) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                elif method == "DELETE":
                    async with session.delete(full_url, params=query_params, headers=request_headers, timeout=req_timeout) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            except aiohttp.ClientResponseError as e:
                return {"error": f"HTTP error: {e.status} {e.message}"}
            except asyncio.TimeoutError:
                return {"error": "Request timed out"}
            except Exception as e:
                return {"error": str(e)}


def json_to_tools_schema(tool_json: Union[Dict[str, Any], List[Dict[str, Any]]]) -> ToolsSchema:
    if isinstance(tool_json, dict):
        tool_json = [tool_json]
    function_schemas = []
    for tool in tool_json:
        # For now, only WebhookTool is supported
        webhook_tool = WebhookTool.from_json(tool)
        function_schemas.append(webhook_tool.to_schema())
    return ToolsSchema(standard_tools=function_schemas)


def build_tools(tools_json: List[Dict[str, Any]]) -> List[Tool]:
    tools = []
    for tool in tools_json:
        if tool["type"] == "webhook":
            tools.append(WebhookTool.from_json(tool))
    return tools