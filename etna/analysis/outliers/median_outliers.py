import math
import typing
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_anomalies_median(
    ts: "TSDataset", in_column: str = "target", window_size: int = 10, alpha: float = 3, index_only: bool = True
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """
    Get point outliers in time series using median model (estimation model-based method).

    Outliers are all points deviating from the median by more than alpha * std,
    where std is the sample standard deviation in the window.

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
    index_only:
        whether to return only outliers indices. If `False` will return outliers series

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}
    """
    outliers_per_segment = {}
    segments = ts.segments
    for seg in segments:
        anomalies: List[int] = []

        segment_df = ts._df[seg].reset_index()
        values = segment_df[in_column].values

        n_iter = math.ceil(len(values) / window_size)
        for i in range(n_iter):
            left_border = i * window_size
            right_border = min(left_border + window_size, len(values))
            med = np.median(values[left_border:right_border])
            std = np.std(values[left_border:right_border])
            diff = np.abs(values[left_border:right_border] - med)
            anomalies.extend(np.where(diff > std * alpha)[0] + left_border)

        if len(anomalies):
            store_values = segment_df.iloc[anomalies]

            if index_only:
                store_values = list(store_values["timestamp"].values)
            else:
                store_values = pd.Series(store_values[in_column].values, index=store_values["timestamp"])

            outliers_per_segment[seg] = store_values

    return outliers_per_segment
