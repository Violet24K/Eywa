"""Shared storage and data-loading tool for time-series MCP servers."""

from io import StringIO
from typing import Any

import pandas as pd
from fastmcp import FastMCP


class TimeSeriesStorage:
    """In-process storage for the currently active time-series task."""

    def __init__(self):
        self.dataframe: pd.DataFrame | None = None
        self.prediction_length: int | None = None


def create_load_time_series_data_tool(mcp: FastMCP, ts_storage: TimeSeriesStorage):
    """Register an internal MCP tool that loads a time-series task into ``ts_storage``."""

    @mcp.tool()
    def load_time_series_data(time_series_task_sample: dict[str, Any]) -> str:
        """INTERNAL TOOL — DO NOT CALL THIS DIRECTLY.

        The Eywa agent will populate the storage at initialization time.
        Planner LLMs should call ``forecast_with_chronos()`` to predict.

        Args:
            time_series_task_sample: A benchmark row with keys ``input`` and
                ``output_size``.
        """
        description = time_series_task_sample.get("description", "<unknown>")
        try:
            ts_storage.dataframe = pd.read_csv(
                StringIO(time_series_task_sample["input"])
            )
            ts_storage.prediction_length = int(time_series_task_sample["output_size"])
            return "Time series data loaded successfully."
        except Exception as e:  # noqa: BLE001 - surfaced back to the MCP client
            print(f"Error loading time series data for {description!r}: {e}")
            return "You should not call this function directly."

    return load_time_series_data
