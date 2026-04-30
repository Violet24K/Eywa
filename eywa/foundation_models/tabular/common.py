"""Shared storage and data-loading tool for tabular MCP servers."""

from io import StringIO
from typing import Any

import pandas as pd
from fastmcp import FastMCP


class TabularStorage:
    """In-process storage for the currently active tabular task."""

    def __init__(self):
        self.dataframe: pd.DataFrame | None = None
        self.task_type: str | None = None
        self.output_size: int | None = None


def create_load_tabular_data_tool(mcp: FastMCP, tabular_storage: TabularStorage):
    """Register an internal MCP tool that loads a tabular task into ``tabular_storage``."""

    @mcp.tool()
    def load_tabular_data(tabular_task_sample: dict[str, Any]) -> str:
        """INTERNAL TOOL — DO NOT CALL THIS DIRECTLY.

        The Eywa agent will populate the storage at initialization time.
        Planner LLMs should call ``tabpfn()`` to make predictions instead.

        Args:
            tabular_task_sample: A benchmark row with keys ``input``, ``task``,
                ``output_size``, and ``description``.
        """
        description = tabular_task_sample.get("description", "<unknown>")
        print(f"Loading tabular data for {description!r}...")
        try:
            tabular_storage.dataframe = pd.read_csv(StringIO(tabular_task_sample["input"]))
            # Normalize task naming so '_' and '-' are interchangeable.
            tabular_storage.task_type = str(tabular_task_sample["task"]).replace("_", "-")
            tabular_storage.output_size = int(tabular_task_sample["output_size"])
            print(f"Tabular data for {description!r} loaded successfully.")
            return f"Tabular data for {description!r} loaded successfully."
        except Exception as e:  # noqa: BLE001 - surfaced back to the MCP client
            print(f"Error loading tabular data for {description!r}: {e}")
            return "You should not call this function directly."

    return load_tabular_data
