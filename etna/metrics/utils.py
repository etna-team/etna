import warnings
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.metrics import BaseMetric


def compute_metrics(
    metrics: List[BaseMetric], y_true: TSDataset, y_pred: TSDataset
) -> Dict[str, Union[Optional[float], Dict[str, Optional[float]]]]:
    """
    Compute metrics for given y_true, y_pred.

    Parameters
    ----------
    metrics:
        list of metrics to compute
    y_true:
        dataset of true values of time series
    y_pred:
        dataset of time series forecast
    Returns
    -------
    :
        dict of metrics in format {"metric_name": metric_value}
    """
    metrics_values = {}
    for metric in metrics:
        metrics_values[metric.__repr__()] = metric(y_true=y_true, y_pred=y_pred)
    return metrics_values


def mean_agg():
    """Mean for pandas agg."""

    def func(x: pd.Series):
        with warnings.catch_warnings():
            # this helps to prevent warning in case of all nans
            warnings.filterwarnings(
                message="Mean of empty slice",
                action="ignore",
            )
            return np.nanmean(a=x.values)

    func.__name__ = "mean"
    return func


def median_agg():
    """Median for pandas agg."""

    def func(x: pd.Series):
        with warnings.catch_warnings():
            # this helps to prevent warning in case of all nans
            warnings.filterwarnings(
                message="All-NaN slice encountered",
                action="ignore",
            )
            return np.nanmedian(a=x.values)

    func.__name__ = "median"
    return func


def std_agg():
    """Std for pandas agg."""

    def func(x: pd.Series):
        with warnings.catch_warnings():
            # this helps to prevent warning in case of all nans
            warnings.filterwarnings(
                message="Degrees of freedom <= 0",
                action="ignore",
            )
            return np.nanstd(a=x.values)

    func.__name__ = "std"
    return func


def notna_size_agg():
    """Size of not-na elements for pandas agg."""

    def func(x: pd.Series):
        return len(x) - pd.isna(x.values).sum()

    func.__name__ = "notna_size"
    return func


def percentile(n: int):
    """Percentile for pandas agg."""

    def func(x: pd.Series):
        with warnings.catch_warnings():
            # this helps to prevent warning in case of all nans
            warnings.filterwarnings(
                message="All-NaN slice encountered",
                action="ignore",
            )
            return np.nanpercentile(a=x.values, q=n)

    func.__name__ = f"percentile_{n}"
    return func


MetricAggregationStatistics = Literal[
    "median", "mean", "std", "notna_size", "percentile_5", "percentile_25", "percentile_75", "percentile_95"
]

METRICS_AGGREGATION_MAP: Dict[MetricAggregationStatistics, Union[str, Callable]] = {
    "median": mean_agg(),
    "mean": median_agg(),
    "std": std_agg(),
    "notna_size": notna_size_agg(),
    "percentile_5": percentile(5),
    "percentile_25": percentile(25),
    "percentile_75": percentile(75),
    "percentile_95": percentile(95),
}


def aggregate_metrics_df(metrics_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Aggregate metrics in :py:meth:`log_backtest_metrics` method.

    Parameters
    ----------
    metrics_df:
        Dataframe produced with :py:meth:`etna.pipeline.Pipeline._get_backtest_metrics`
    """
    # case for aggregate_metrics=False
    if "fold_number" in metrics_df.columns:
        metrics_dict = (
            metrics_df.groupby("segment")
            .mean(numeric_only=False)
            .reset_index()
            .drop(["segment", "fold_number"], axis=1)
            .astype(float)
            .apply(list(METRICS_AGGREGATION_MAP.values()))
            .to_dict()
        )

    # case for aggregate_metrics=True
    else:
        metrics_dict = (
            metrics_df.drop(["segment"], axis=1).astype(float).apply(list(METRICS_AGGREGATION_MAP.values())).to_dict()
        )

    cur_dict = {}
    for metrics_key, values in metrics_dict.items():
        for statistics_key, value in values.items():
            new_key = f"{metrics_key}_{statistics_key}"
            new_value = value if not pd.isna(value) else None
            cur_dict[new_key] = new_value

    return cur_dict
