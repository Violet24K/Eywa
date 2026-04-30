"""Tabular classification and regression metrics."""

from io import StringIO

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from eywabench.eval.timeseries import compute_maape_nonzero, compute_smape


def eval_tabular_classification(
    label_array: np.ndarray, predicted_array: np.ndarray
) -> float:
    """Plain accuracy in [0, 1]."""
    return float(accuracy_score(label_array, predicted_array))


def eval_tabular_regression(
    label_array: np.ndarray, predicted_array: np.ndarray
) -> float:
    """Combine sMAPE and MAAPE into a utility score in [0, 1]."""
    smape = compute_smape(label_array, predicted_array)
    maape = compute_maape_nonzero(label_array, predicted_array)
    return float(1 - ((smape / 2 + maape / np.pi * 2) / 2))


def eval_tabular_string_format(
    label_string: str, predicted_string: str, task: str
) -> float:
    """Score a tabular prediction provided as CSV-or-list-string against CSV labels."""
    label_array = pd.read_csv(StringIO(label_string)).values.flatten()
    if predicted_string.startswith("["):
        predicted_array = np.array(eval(predicted_string))
    else:
        predicted_array = pd.read_csv(StringIO(predicted_string)).values.flatten()

    if task == "tabular-classification":
        return eval_tabular_classification(label_array, predicted_array)
    return eval_tabular_regression(label_array, predicted_array)
