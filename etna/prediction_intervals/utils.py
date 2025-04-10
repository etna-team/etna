from typing import Optional

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.pipeline import BasePipeline


def residuals_matrices(
    pipeline: BasePipeline, ts: TSDataset, n_folds: int = 5, stride: Optional[int] = None
) -> np.ndarray:
    """Estimate residuals matrices with backtest.

    Parameters
    ----------
    pipeline:
        Pipeline for residuals estimation.
    ts:
        Dataset to estimate residuals.
    n_folds:
        Number of folds for backtest.
    stride:
        Number of points between folds. By default, is set to ``horizon``.

    Returns
    -------
    :
        Residuals matrices for each segment. Array with shape: ``(n_folds, horizon, n_segments)``.
    """
    if n_folds <= 0:
        raise ValueError("Parameter `n_folds` must be positive!")

    if stride is not None and stride <= 0:
        raise ValueError("Parameter `stride` must be positive!")

    list_backtest_forecasts = pipeline.get_historical_forecasts(ts=ts, n_folds=n_folds, stride=stride)
    backtest_forecasts = pd.concat([forecast.to_pandas() for forecast in list_backtest_forecasts], axis=0)

    residuals = backtest_forecasts.loc[:, pd.IndexSlice[:, "target"]] - ts[backtest_forecasts.index, :, "target"]

    # shape: (n_folds, horizon, n_segments)
    residual_matrices = residuals.values.reshape((-1, pipeline.horizon, len(ts.segments)))
    return residual_matrices
