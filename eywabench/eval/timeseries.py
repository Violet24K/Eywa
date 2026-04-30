"""Time-series forecasting metrics."""

from io import StringIO

import numpy as np
import pandas as pd


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-2) -> float:
    """Symmetric Mean Absolute Percentage Error in [0, 2]."""
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom))


def compute_maape_nonzero(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-2
) -> float:
    """Mean Arctangent Absolute Percentage Error in [0, pi/2].

    Indices where ``|y_true| <= eps`` are skipped to keep the metric well
    defined. Returns ``0`` if every index is skipped.
    """
    valid = np.abs(y_true) > eps
    if valid.sum() == 0:
        return 0.0
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.arctan(np.abs((y_true[valid] - y_pred[valid]) / denom[valid]))))


def eval_timeseries(future_df: pd.DataFrame, forecasted_df: pd.DataFrame) -> float:
    """Combine sMAPE and MAAPE into a utility score in [0, 1] (higher is better)."""
    y_true = future_df["value"].values
    y_pred = forecasted_df["value"].values
    smape = compute_smape(y_true, y_pred)
    maape = compute_maape_nonzero(y_true, y_pred)
    return float(1 - ((smape / 2 + maape / np.pi * 2) / 2))


def eval_timeseries_string_format(label_string: str, forecasted_string: str) -> float:
    """Convenience wrapper that parses CSV strings before scoring."""
    future_df = pd.read_csv(StringIO(label_string))
    forecasted_df = pd.read_csv(StringIO(forecasted_string))
    return eval_timeseries(future_df, forecasted_df)
