import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.tsatools import freq_to_period

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def _sliding_window(x: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Prepare windows of 1-d data, strided by given parameters."""
    if window_size <= 0:
        raise ValueError("Window size must be positive integer!")

    if stride < 1:
        raise ValueError("Stride must be integer greater or equal to 1!")

    # get all sliding windows views
    all_windows = sliding_window_view(x[::-1], window_size)

    # select only windows, that match given stride
    strided_windows = all_windows[::stride, ::-1]

    # reverse back to match the original order
    return strided_windows[::-1]


def sliding_window_decorator(func: Callable) -> Callable:
    """Decorate function to run on windows of 1-d data."""

    @functools.wraps(func)
    def wrapper(
        series: pd.Series, *, window_size: int, stride: int = 1, return_indices: bool = True, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make computation on sliding window with given stride.

        Parameters
        ----------
        series:
            original series.
        window_size:
            size of the window to make computations on.
        stride:
            offset between neighboring windows.
        return_indices:
            whether to return original indices along with the results.
        **kwargs:
            additional arguments for the function.

        Returns
        -------
        :
            arrays with the results of applying function on the sliding windows.
        """
        indices = np.arange(series.size)
        indices_matrix = _sliding_window(x=indices, window_size=window_size, stride=stride)

        apply_func = functools.partial(func, series, **kwargs)

        results = np.apply_along_axis(apply_func, 1, indices_matrix)

        if return_indices:
            return results, indices_matrix

        else:
            return results

    return wrapper


def _stl_decompose(
    series: pd.Series, period: Optional[int] = None, trend: bool = False, seasonality: bool = False, **kwargs
) -> pd.Series:
    """
    Estimate seasonal and trend components and remove them from the series.

    Parameters
    ----------
    series:
        series for detrending and seasonality removal.
    period:
        periodicity of the sequence.
    trend:
        whether to remove trend from the series.
    seasonality:
        whether to remove seasonality from the series
    **kwargs:
        other parameters for decompositions. See :py:class:`statsmodels.tsa.seasonal.STL`

    Returns
    -------
    :
        series with removed seasonality/trend.
    """
    if not (trend or seasonality):
        raise ValueError("At least one component must be set!")

    if period is None:
        freq = getattr(series.index, "inferred_freq", None)
        if freq is None:
            raise ValueError("Series must have inferable frequency to autodetect period for STL!")

        period = freq_to_period(freq)

    stl_res = STL(endog=series, period=period, **kwargs).fit()

    if trend:
        series = series - stl_res.trend

    if seasonality:
        series = series - stl_res.seasonal

    return series


def _outliers_per_segment(
    df: pd.DataFrame, func: Callable, index_only: bool = True
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """Run estimation function for each segment."""
    outliers_per_segment = {}
    for segment in df.columns:
        series = df[segment]

        series = series.loc[series.first_valid_index() :]

        if np.any(series.isna()):
            raise ValueError(f"Segment `{segment}` contains missing values!")

        mask, indices = func(series)

        # set index as outlier if it was marked as such at least in one window
        outlier_indices = np.unique(indices[mask].reshape(-1))

        if len(outlier_indices) > 0:
            if index_only:
                outliers = list(series.index[outlier_indices].values)
            else:
                outliers = series.iloc[outlier_indices]

            outliers_per_segment[segment] = outliers

    return outliers_per_segment


@sliding_window_decorator
def iqr_method(
    series: pd.Series,
    indices: np.ndarray,
    iqr_scale: float = 1.5,
    period: Optional[int] = None,
    trend: bool = False,
    seasonality: bool = False,
    stl_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Estimate anomalies using IQR statistics.

    Parameters
    ----------
    series:
        original series for the estimation.
    indices:
        which observations use for the estimation.
    iqr_scale:
        scaling parameter of the estimated interval.
    period:
        periodicity of the sequence for STL.
    trend:
        whether to remove trend from the series.
    seasonality:
        whether to remove seasonality from the series
    stl_params:
        other parameters for STL. See :py:class:`statsmodels.tsa.seasonal.STL`

    Returns
    -------
    :
        binary mask for each observation, indicating if it was estimated as an anomaly.
    """
    if iqr_scale <= 0:
        raise ValueError("Scaling parameter must be positive!")

    window = series.iloc[indices]

    if trend or seasonality:
        if stl_params is None:
            stl_params = {}

        window = _stl_decompose(series=window, period=period, trend=trend, seasonality=seasonality, **stl_params)

    window = window.values

    first = np.quantile(window, q=0.25)
    third = np.quantile(window, q=0.75)

    iqr = iqr_scale * (third - first)

    maxval = third + iqr
    minval = first - iqr

    anom_mask = (window > maxval) | (window < minval)

    return anom_mask


def get_anomalies_iqr(
    ts: "TSDataset",
    in_column: str = "target",
    window_size: int = 10,
    stride: int = 1,
    iqr_scale: float = 1.5,
    trend: bool = False,
    seasonality: bool = False,
    period: Optional[int] = None,
    stl_params: Optional[Dict[str, Any]] = None,
    index_only: bool = True,
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """
    Get point outliers in time series using IQR statistics, estimated on a rolling window.

    Outliers are all points that fall outside the estimated interval.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    window_size:
        number of points in the window
    stride:
        offset between neighboring windows.
    iqr_scale:
        scaling parameter of the estimated interval.
    trend:
        whether to remove trend from the series.
    seasonality:
        whether to remove seasonality from the series
    period:
        periodicity of the sequence for STL.
    stl_params:
        other parameters for STL. See :py:class:`statsmodels.tsa.seasonal.STL`
    index_only:
        whether to return only outliers indices. If `False` will return outliers series

    Returns
    -------
    :
        mapping from segment names to corresponding outliers.
    """
    df = ts[..., in_column]
    df = df.droplevel(level="feature", axis=1)

    detection_func = functools.partial(
        iqr_method,
        window_size=window_size,
        stride=stride,
        iqr_scale=iqr_scale,
        trend=trend,
        seasonality=seasonality,
        period=period,
        stl_params=stl_params,
    )

    return _outliers_per_segment(df=df, func=detection_func, index_only=index_only)


@sliding_window_decorator
def mad_method(
    series: pd.Series,
    indices: np.ndarray,
    mad_scale: float = 1.5,
    period: Optional[int] = None,
    trend: bool = False,
    seasonality: bool = False,
    stl_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Estimate anomalies using MAD statistics.
    Parameters
    ----------
    series:
        original series for the estimation.
    indices:
        which observations use for the estimation.
    mad_scale:
        scaling parameter of the estimated interval.
    period:
        periodicity of the sequence for STL.
    trend:
        whether to remove trend from the series.
    seasonality:
        whether to remove seasonality from the series
    stl_params:
        other parameters for STL. See :py:class:`statsmodels.tsa.seasonal.STL`
    Returns
    -------
    :
        binary mask for each observation, indicating if it was estimated as an anomaly.
    """
    if mad_scale <= 0:
        raise ValueError("Scaling parameter must be positive!")

    window = series.iloc[indices]

    if trend or seasonality:
        if stl_params is None:
            stl_params = {}

        window = _stl_decompose(series=window, period=period, trend=trend, seasonality=seasonality, **stl_params)

    mad = median_abs_deviation(window)
    median = np.median(window)

    anom_mask = np.abs(window - median) > mad * mad_scale

    return anom_mask


def get_anomalies_mad(
    ts: "TSDataset",
    in_column: str = "target",
    window_size: int = 10,
    stride: int = 1,
    mad_scale: float = 3,
    trend: bool = False,
    seasonality: bool = False,
    period: Optional[int] = None,
    stl_params: Optional[Dict[str, Any]] = None,
    index_only: bool = True,
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """
    Get point outliers in time series using median absolute deviation.
    Detects outliers in a row that fall out of range: [median - mad_scale * mad; median + mad_scale * mad]

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    window_size:
        number of points in the window
    stride:
        offset between neighboring windows.
    mad_scale:
        scaling parameter of the estimated interval.
    trend:
        whether to remove trend from the series.
    seasonality:
        whether to remove seasonality from the series
    period:
        periodicity of the sequence for STL.
    stl_params:
        other parameters for STL. See :py:class:`statsmodels.tsa.seasonal.STL`
    index_only:
        whether to return only outliers indices. If `False` will return outliers series

    Returns
    -------
    :
       dict of outliers in format {segment: [outliers_timestamps]}
    """
    df = ts[..., in_column]
    df = df.droplevel(level="feature", axis=1)

    detection_func = functools.partial(
        mad_method,
        window_size=window_size,
        stride=stride,
        mad_scale=mad_scale,
        trend=trend,
        seasonality=seasonality,
        period=period,
        stl_params=stl_params,
    )

    return _outliers_per_segment(df=df, func=detection_func, index_only=index_only)
