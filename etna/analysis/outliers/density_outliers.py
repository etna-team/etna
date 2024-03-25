from enum import Enum
from itertools import islice
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def absolute_difference_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate distance for :py:func:`get_anomalies_density` function by taking absolute value of difference.

    Parameters
    ----------
    x:
        first value
    y:
        second value

    Returns
    -------
    result: np.ndarray
        absolute difference between values
    """
    return np.abs(x - y)


class DistanceFunction(str, Enum):
    """Enum for points distance measure functions."""

    absolute_difference = "absolute_difference"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} seasonality allowed"
        )

    def get_callable(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if self.value == DistanceFunction.absolute_difference:
            return absolute_difference_distance
        else:
            raise ValueError("Invalid distance function!")


def get_segment_density_outliers_indices(
    series: np.ndarray,
    window_size: int = 7,
    distance_threshold: float = 10,
    n_neighbors: int = 3,
    distance_func: Union[Literal["absolute_difference"], Callable[[float, float], float]] = "absolute_difference",
) -> List[int]:
    """Get indices of outliers for one series.

    Parameters
    ----------
    series:
        array to find outliers in
    window_size:
        size of window
    distance_threshold:
        if distance between two items in the window is less than threshold those items are supposed to be close to each other
    n_neighbors:
        min number of close items that item should have not to be outlier
    distance_func:
        distance function. If a string is specified, a corresponding vectorized implementation will be used. Custom callable
        will be used as a scalar function, which will result in worse performance.

    Returns
    -------
    :
        list of outliers' indices
    """
    idxs = np.arange(len(series))
    start_idxs = np.maximum(0, idxs - window_size)
    end_idxs = np.maximum(0, np.minimum(idxs, len(series) - window_size)) + 1

    max_shifts: np.ndarray = end_idxs - start_idxs

    if isinstance(distance_func, str):
        dist_func = DistanceFunction(distance_func).get_callable()

        def _closeness_func(x, start, stop, y):
            return (dist_func(x[start:stop], y) < distance_threshold).astype(int)

    else:

        def _closeness_func(x, start, stop, y):
            return [int(distance_func(elem, y) < distance_threshold) for elem in islice(x, start, stop)]

    outliers_indices = []
    for idx, item, start_idx, max_shift in zip(idxs, series, start_idxs, max_shifts):
        # compute which neighbours are close to the element in the given windows
        closeness = _closeness_func(
            x=series,
            start=start_idx,
            stop=start_idx + window_size + max_shift - 1,
            y=item,
        )

        # compute number of close neighbours before index
        num_close = np.cumsum(closeness)

        outlier = True
        for shift in range(max_shift):
            # number of neighbours in particular window
            num_in_window = num_close[-max_shift + shift] - num_close[shift]
            if (start_idx + shift) != idx:
                # subtract current element if it is not on the window border
                num_in_window += closeness[shift] - 1

            if num_in_window >= n_neighbors:
                outlier = False
                break

        if outlier:
            outliers_indices.append(idx)

    return outliers_indices


def get_anomalies_density(
    ts: "TSDataset",
    in_column: str = "target",
    window_size: int = 15,
    distance_coef: float = 3,
    n_neighbors: int = 3,
    distance_func: Union[Literal["absolute_difference"], Callable[[float, float], float]] = "absolute_difference",
    index_only: bool = True,
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """Compute outliers according to density rule.

    For each element in the series build all the windows of size ``window_size`` containing this point.
    If any of the windows contains at least ``n_neighbors`` that are closer than ``distance_coef * std(series)``
    to target point according to ``distance_func`` target point is not an outlier.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    window_size:
        size of windows to build
    distance_coef:
        factor for standard deviation that forms distance threshold to determine points are close to each other
    n_neighbors:
        min number of close neighbors of point not to be outlier
    distance_func:
        distance function. If a string is specified, a corresponding vectorized implementation will be used. Custom callable
        will be used as a scalar function, which will result in worse performance.
    index_only:
        whether to return only outliers indices. If `False` will return outliers series

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}

    Notes
    -----
    It is a variation of distance-based (index) outlier detection method adopted for timeseries.
    """
    outliers_per_segment = {}

    segments_df = ts[..., in_column].droplevel("feature", axis=1)
    stds = np.nanstd(segments_df.values, axis=0)

    for series_std, (segment, series_df) in zip(stds, segments_df.items()):
        # TODO: dropna() now is responsible for removing nan-s at the end of the sequence and in the middle of it
        #   May be error or warning should be raised in this case
        series = series_df.dropna()

        if series_std > 0:
            outliers_idxs = get_segment_density_outliers_indices(
                series=series.values,
                window_size=window_size,
                distance_threshold=distance_coef * series_std,
                n_neighbors=n_neighbors,
                distance_func=distance_func,
            )

            if len(outliers_idxs):
                if index_only:
                    store_values = list(series.index.values[outliers_idxs])

                else:
                    store_values = series.iloc[outliers_idxs]

                outliers_per_segment[segment] = store_values

    return outliers_per_segment


__all__ = ["get_anomalies_density", "absolute_difference_distance"]
