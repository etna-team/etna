import math
import typing

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

if typing.TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_anomalies_median(
    ts: "TSDataset", in_column: str = "target", window_size: int = 10, alpha: float = 3
) -> Tuple[Dict[str, List[pd.Timestamp]], Dict[str, pd.Series]]:
    """
    Get point outliers in time series using median model (estimation model-based method).

    Outliers are all points deviating from the median by more than alpha * std,
    where std is the sample variance in the window.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    window_size:
        number of points in the window
    alpha:
        coefficient for determining the threshold

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}
    """
    segments = np.array(sorted(ts.segments))
    values = ts.df.loc[:, pd.IndexSlice[segments, in_column]].values
    timestamps = ts.index.values

    anomalies_rows, anomalies_cols = [], []
    n_iter = math.ceil(len(timestamps) / window_size)
    for i in range(n_iter):
        left_border = i * window_size
        right_border = min(left_border + window_size, len(values))
        med = np.median(values[left_border:right_border])
        std = np.std(values[left_border:right_border])
        diff = np.abs(values[left_border:right_border] - med)
        row, col = np.nonzero(diff > std * alpha)
        row += left_border
        anomalies_rows.extend(row)
        anomalies_cols.extend(col)

    anomalies_df = pd.DataFrame({
        "timestamp": timestamps[anomalies_rows],
        "segment": segments[anomalies_cols],
        "target": values[anomalies_rows, anomalies_cols]
    })

    outliers_per_segment, outliers_values_per_segment = defaultdict(list), dict()
    for segment, df in anomalies_df.groupby("segment"):
        outliers_per_segment[segment] = df["timestamp"].values
        outliers_values_per_segment[segment] = pd.Series(index=df["timestamp"].values, data=df["target"].values)

    return outliers_per_segment, outliers_values_per_segment, anomalies_rows, anomalies_cols
