"""Chronos (via AutoGluon) as an MCP server.

The agent calls ``forecast_with_chronos`` with no arguments; the input
series and prediction length are pre-loaded into ``TimeSeriesStorage``
through the internal ``load_time_series_data`` tool.
"""

import argparse

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from fastmcp import FastMCP

from eywa.foundation_models.time_series.common import (
    TimeSeriesStorage,
    create_load_time_series_data_tool,
)
from eywa.utils.path import chronos_mcp_server_port


mcp = FastMCP("chronos_mcp_server", json_response=True)
ts_storage = TimeSeriesStorage()
create_load_time_series_data_tool(mcp, ts_storage)


@mcp.tool()
def forecast_with_chronos() -> str:
    """Forecast the loaded time series using AutoGluon's Chronos2 preset.

    Returns:
        The mean forecast as a stringified pandas Series.
    """
    target = "value"
    print(
        f"Forecasting with Chronos: prediction_length={ts_storage.prediction_length}, "
        f"target={target}."
    )

    df_data = ts_storage.dataframe.copy()
    if "id" not in df_data.columns:
        df_data["id"] = 0
    df_data["value"] = df_data["value"].astype(float)

    data = TimeSeriesDataFrame.from_data_frame(
        df_data, id_column="id", timestamp_column="timestamp"
    )
    predictor = TimeSeriesPredictor(
        prediction_length=ts_storage.prediction_length, target=target
    ).fit(data, presets="chronos2")
    predictions = predictor.predict(data)

    print(f"Predictions: {predictions}")
    predictions_reset = predictions.reset_index()
    ts_storage.foundation_model_forecasting = pd.DataFrame(
        {
            "timestamp": predictions_reset["timestamp"],
            "value": predictions_reset["mean"],
        }
    )
    return predictions["mean"].to_string()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=chronos_mcp_server_port)
    cli_args = parser.parse_args()
    mcp.run(transport="streamable-http", port=cli_args.port)
