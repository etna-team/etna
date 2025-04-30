import warnings
from reprlib import repr
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.metrics import BaseMetric
from etna.metrics.base import MetricMissingMode


class BaseIntervalsMetricWithMissingHandling(BaseMetric):
    """Base class for metrics for prediction intervals with missing handling."""

    def __init__(
        self,
        quantiles: Optional[Tuple[float, float]] = None,
        mode: str = "per-segment",
        upper_name: Optional[str] = None,
        lower_name: Optional[str] = None,
        missing_mode: str = "error",
        **kwargs,
    ):
        """Init metric.

        Parameters
        ----------
        quantiles:
            lower and upper quantiles
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        upper_name:
            name of column with upper border of the interval
        lower_name:
            name of column with lower border of the interval
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode)

        self.missing_mode = missing_mode
        self._missing_mode_enum = MetricMissingMode(missing_mode)

        if (lower_name is None) ^ (upper_name is None):
            raise ValueError("Both `lower_name` and `upper_name` must be set if using names to specify borders!")

        if not (quantiles is None or lower_name is None):
            raise ValueError(
                "Both `quantiles` and border names are specified. Use only one way to set interval borders!"
            )

        if quantiles is not None and len(quantiles) != 2:
            raise ValueError(f"Expected tuple with two values for `quantiles` parameter, got {len(quantiles)}")

        # default behavior
        if quantiles is None and lower_name is None:
            quantiles = (0.025, 0.975)

        self.quantiles = sorted(quantiles if quantiles is not None else tuple())
        self.upper_name = upper_name
        self.lower_name = lower_name
        self.kwargs = kwargs

    @staticmethod
    def _validate_tsdataset_intervals(
        ts: TSDataset, quantiles: Sequence[float], upper_name: Optional[str], lower_name: Optional[str]
    ) -> None:
        """Check if intervals borders presented in ``y_pred``."""
        ts_intervals = set(ts.prediction_intervals_names)

        borders_set = {upper_name, lower_name}
        borders_presented = borders_set.issubset(ts_intervals)

        quantiles_set = {f"target_{quantile:.4g}" for quantile in quantiles}
        quantiles_presented = quantiles_set.issubset(ts_intervals)
        quantiles_presented &= len(quantiles_set) > 0

        if upper_name is not None and lower_name is not None:
            if not borders_presented:
                raise ValueError("Provided intervals borders names must be in dataset!")

            else:
                missing_per_segment = ts._df.loc[:, pd.IndexSlice[:, list(borders_set)]].isna().any()
                if missing_per_segment.any():
                    raise ValueError(
                        "Provided intervals borders contain missing values! "
                        f"Series with missing values {repr(missing_per_segment[missing_per_segment].index.tolist())}"
                    )

        else:
            if not quantiles_presented:
                raise ValueError("All quantiles must be presented in the dataset!")

            else:
                missing_per_segment = ts._df.loc[:, pd.IndexSlice[:, list(quantiles_set)]].isna().any()
                if missing_per_segment.any():
                    raise ValueError(
                        "Quantiles contain missing values! "
                        f"Series with missing values {repr(missing_per_segment[missing_per_segment].index.tolist())}"
                    )

    def _validate_nans(self, y_true: TSDataset, y_pred: TSDataset):
        """Check that ``y_true`` and ``y_pred`` doesn't have NaNs depending on ``missing_mode``.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            If there are NaNs in ``y_true`` or ``y_pred``
        """
        df_true = y_true._df.loc[:, pd.IndexSlice[:, "target"]]
        df_pred = y_pred._df.loc[:, pd.IndexSlice[:, "target"]]

        df_true_isna_sum = df_true.isna().sum()
        if self._missing_mode_enum is MetricMissingMode.error and (df_true_isna_sum > 0).any():
            error_segments = set(df_true_isna_sum[df_true_isna_sum > 0].index.droplevel("feature").tolist())
            raise ValueError(f"There are NaNs in y_true! Segments with NaNs: {repr(error_segments)}.")

        df_pred_isna_sum = df_pred.isna().sum()
        if (df_pred_isna_sum > 0).any():
            error_segments = set(df_pred_isna_sum[df_pred_isna_sum > 0].index.droplevel("feature").tolist())
            raise ValueError(f"There are NaNs in y_pred Segments with NaNs: {repr(error_segments)}.")

    @staticmethod
    def _macro_average(metrics_per_segments: Dict[str, Optional[float]]) -> Optional[float]:
        """
        Compute macro averaging of metrics over segment.

        None values are ignored during computation.

        Parameters
        ----------
        metrics_per_segments: dict of {segment: metric_value} for segments to aggregate

        Returns
        -------
        :
            aggregated value of metric
        """
        with warnings.catch_warnings():
            # this helps to prevent warning in case of all nans
            warnings.filterwarnings(
                message="Mean of empty slice",
                action="ignore",
            )
            # dtype=float is used to cast None to np.nan
            value = np.nanmean(np.fromiter(metrics_per_segments.values(), dtype=float)).item()
        if np.isnan(value):
            return None
        else:
            return value

    def _validate_base_interval(self, y_true: TSDataset, y_pred: TSDataset):
        """
        Check that ``y_true`` and ``y_pred`` pass all validations for interval metrics with missing handling.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset
        """
        self._validate_base(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)
        self._validate_tsdataset_intervals(
            ts=y_pred, quantiles=self.quantiles, upper_name=self.upper_name, lower_name=self.lower_name
        )


class Coverage(BaseIntervalsMetricWithMissingHandling):
    """Coverage metric for prediction intervals - precenteage of samples in the interval ``[lower quantile, upper quantile]``.

    .. math::
        Coverage(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n}{[ y\_true_i \\ge y\_pred_i^{lower\_quantile}] * [y\_true_i \\le y\_pred_i^{upper\_quantile}] }}{n}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    Works just if ``quantiles`` presented in ``y_pred``

    When ``quantiles``, ``upper_name`` and ``lower_name`` all set to ``None`` then 0.025 and 0.975 quantiles will be used.
    """

    def __init__(
        self,
        quantiles: Optional[Tuple[float, float]] = None,
        mode: str = "per-segment",
        upper_name: Optional[str] = None,
        lower_name: Optional[str] = None,
        missing_mode: str = "error",
        **kwargs,
    ):
        """Init metric.

        Parameters
        ----------
        quantiles:
            lower and upper quantiles
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        upper_name:
            name of column with upper border of the interval
        lower_name:
            name of column with lower border of the interval
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        super().__init__(
            quantiles=quantiles,
            mode=mode,
            upper_name=upper_name,
            lower_name=lower_name,
            missing_mode=missing_mode,
            **kwargs,
        )

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[Optional[float], Dict[str, Optional[float]]]:
        """
        Compute metric's value with y_true and y_pred.

        Notes
        -----
        Note that if y_true and y_pred are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
            metric's value aggregated over segments or not (depends on mode)
        """
        self._validate_base_interval(y_true=y_true, y_pred=y_pred)

        if self.upper_name is not None:
            lower_border = self.lower_name
            upper_border = self.upper_name

        else:
            lower_border = f"target_{self.quantiles[0]:.4g}"
            upper_border = f"target_{self.quantiles[1]:.4g}"

        df_true = y_true._df.loc[:, pd.IndexSlice[:, "target"]].sort_index(axis=1)

        intervals_df: pd.DataFrame = y_pred.get_prediction_intervals()
        df_pred_lower = intervals_df.loc[:, pd.IndexSlice[:, lower_border]].sort_index(axis=1)
        df_pred_upper = intervals_df.loc[:, pd.IndexSlice[:, upper_border]].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        upper_quantile_flag = df_true.values <= df_pred_upper.values
        lower_quantile_flag = df_true.values >= df_pred_lower.values

        nan_mask = np.isnan(df_true.values) | np.isnan(df_pred_upper.values) | np.isnan(df_pred_lower.values)
        in_bounds = (upper_quantile_flag * lower_quantile_flag).astype(float)
        in_bounds[nan_mask] = np.NaN

        with warnings.catch_warnings():
            warnings.filterwarnings(
                message="Mean of empty slice",
                action="ignore",
            )
            values = np.nanmean(in_bounds, axis=0)

        metrics_per_segment = dict(zip(segments, (None if np.isnan(x) else x for x in values)))

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> None:
        """Whether higher metric value is better."""
        return None


class Width(BaseIntervalsMetricWithMissingHandling):
    """Mean width of prediction intervals.

    .. math::
        Width(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n}\\mid y\_pred_i^{upper\_quantile} - y\_pred_i^{lower\_quantile} \\mid}{n}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    Works just if quantiles presented in ``y_pred``.

    When ``quantiles``, ``upper_name`` and ``lower_name`` all set to ``None`` then 0.025 and 0.975 quantiles will be used.
    """

    def __init__(
        self,
        quantiles: Optional[Tuple[float, float]] = None,
        mode: str = "per-segment",
        upper_name: Optional[str] = None,
        lower_name: Optional[str] = None,
        missing_mode: str = "error",
        **kwargs,
    ):
        """Init metric.

        Parameters
        ----------
        quantiles:
            lower and upper quantiles
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        upper_name:
            name of column with upper border of the interval
        lower_name:
            name of column with lower border of the interval
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        super().__init__(
            quantiles=quantiles,
            mode=mode,
            upper_name=upper_name,
            lower_name=lower_name,
            missing_mode=missing_mode,
            **kwargs,
        )

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[Optional[float], Dict[str, Optional[float]]]:
        """
        Compute metric's value with y_true and y_pred.

        Notes
        -----
        Note that if y_true and y_pred are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
            metric's value aggregated over segments or not (depends on mode)
        """
        self._validate_base_interval(y_true=y_true, y_pred=y_pred)

        if self.upper_name is not None:
            lower_border = self.lower_name
            upper_border = self.upper_name

        else:
            lower_border = f"target_{self.quantiles[0]:.4g}"
            upper_border = f"target_{self.quantiles[1]:.4g}"

        df_true = y_true._df.loc[:, pd.IndexSlice[:, "target"]].sort_index(axis=1)

        intervals_df: pd.DataFrame = y_pred.get_prediction_intervals()
        df_pred_lower = intervals_df.loc[:, pd.IndexSlice[:, lower_border]].sort_index(axis=1)
        df_pred_upper = intervals_df.loc[:, pd.IndexSlice[:, upper_border]].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                message="Mean of empty slice",
                action="ignore",
            )
            values = np.nanmean(np.abs(df_pred_upper.values - df_pred_lower.values), axis=0)

        metrics_per_segment = dict(zip(segments, (None if np.isnan(x) else x for x in values)))

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


__all__ = ["Coverage", "Width"]
