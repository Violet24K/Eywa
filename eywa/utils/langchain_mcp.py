"""Helpers for invoking tools served by an MCP server through LangChain."""

from typing import Any

from langchain_core.tools import StructuredTool


async def invoke_tool(tools: list[StructuredTool], tool_name: str, args: dict) -> Any:
    """Invoke a named tool from a list of LangChain ``StructuredTool`` objects.

    Args:
        tools: Tools resolved from an MCP client.
        tool_name: Name of the tool to call.
        args: Keyword arguments forwarded to the tool.

    Returns:
        Whatever the tool returns, or ``None`` if no tool matched.
    """
    for tool in tools:
        if tool.name == tool_name:
            return await tool.ainvoke(args)
    print(f"Tool {tool_name!r} not found.")
    return None
