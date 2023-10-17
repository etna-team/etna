from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode
from etna.metrics.functional_metrics import ArrayLike


def dummy(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    return np.nan


class _IntervalsMetricMixin:
    def _validate_tsdataset_intervals(
        self, ts: TSDataset, quantiles: Sequence[float], upper_name: Optional[str], lower_name: Optional[str]
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
            if not quantiles_presented:
                raise ValueError("All quantiles must be presented in the dataset!")


class Coverage(Metric, _IntervalsMetricMixin):
    """Coverage metric for prediction intervals - precenteage of samples in the interval ``[lower quantile, upper quantile]``.

    .. math::
        Coverage(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}{[ y\_true_i \\ge y\_pred_i^{lower\_quantile}] * [y\_true_i \\le y\_pred_i^{upper\_quantile}] }}{n}

    Notes
    -----
    Works just if ``quantiles`` presented in ``y_pred``

    When ``quantiles``, ``upper_name`` and ``lower_name`` all set to ``None`` then 0.025 and 0.975 quantiles will be used.
    """

    def __init__(
        self,
        quantiles: Optional[Tuple[float, float]] = None,
        mode: str = MetricAggregationMode.per_segment,
        upper_name: Optional[str] = None,
        lower_name: Optional[str] = None,
        **kwargs,
    ):
        """Init metric.

        Parameters
        ----------
        quantiles:
            lower and upper quantiles
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        upper_name:
            name of column with upper border of the interval
        lower_name:
            name of column with lower border of the interval
        kwargs:
            metric's computation arguments
        """
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

        super().__init__(mode=mode, metric_fn=dummy, **kwargs)
        self.quantiles = sorted(quantiles if quantiles is not None else tuple())
        self.upper_name = upper_name
        self.lower_name = lower_name

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
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
        self._validate_segments(y_true=y_true, y_pred=y_pred)
        self._validate_target_columns(y_true=y_true, y_pred=y_pred)
        self._validate_index(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)
        self._validate_tsdataset_intervals(
            ts=y_pred, quantiles=self.quantiles, lower_name=self.lower_name, upper_name=self.upper_name
        )

        if self.upper_name is not None:
            lower_border = self.lower_name
            upper_border = self.upper_name

        else:
            lower_border = f"target_{self.quantiles[0]:.4g}"
            upper_border = f"target_{self.quantiles[1]:.4g}"

        df_true = y_true[:, :, "target"].sort_index(axis=1)

        intervals_df: pd.DataFrame = y_pred.get_prediction_intervals()
        df_pred_lower = intervals_df.loc[:, pd.IndexSlice[:, lower_border]].sort_index(axis=1)
        df_pred_upper = intervals_df.loc[:, pd.IndexSlice[:, upper_border]].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        upper_quantile_flag = df_true.values <= df_pred_upper.values
        lower_quantile_flag = df_true.values >= df_pred_lower.values
        values = np.mean(upper_quantile_flag * lower_quantile_flag, axis=0)
        metrics_per_segment = dict(zip(segments, values))

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> None:
        """Whether higher metric value is better."""
        return None


class Width(Metric, _IntervalsMetricMixin):
    """Mean width of prediction intervals.

    .. math::
        Width(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}\\mid y\_pred_i^{upper\_quantile} - y\_pred_i^{lower\_quantile} \\mid}{n}

    Notes
    -----
    Works just if quantiles presented in ``y_pred``.

    When ``quantiles``, ``upper_name`` and ``lower_name`` all set to ``None`` then 0.025 and 0.975 quantiles will be used.
    """

    def __init__(
        self,
        quantiles: Optional[Tuple[float, float]] = None,
        mode: str = MetricAggregationMode.per_segment,
        upper_name: Optional[str] = None,
        lower_name: Optional[str] = None,
        **kwargs,
    ):
        """Init metric.

        Parameters
        ----------
        quantiles:
            lower and upper quantiles
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        upper_name:
            name of column with upper border of the interval
        lower_name:
            name of column with lower border of the interval
        kwargs:
            metric's computation arguments
        """
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

        super().__init__(mode=mode, metric_fn=dummy, **kwargs)
        self.quantiles = sorted(quantiles if quantiles is not None else tuple())
        self.upper_name = upper_name
        self.lower_name = lower_name

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
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
        self._validate_segments(y_true=y_true, y_pred=y_pred)
        self._validate_target_columns(y_true=y_true, y_pred=y_pred)
        self._validate_index(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)
        self._validate_tsdataset_intervals(
            ts=y_pred, quantiles=self.quantiles, lower_name=self.lower_name, upper_name=self.upper_name
        )

        if self.upper_name is not None:
            lower_border = self.lower_name
            upper_border = self.upper_name

        else:
            lower_border = f"target_{self.quantiles[0]:.4g}"
            upper_border = f"target_{self.quantiles[1]:.4g}"

        df_true = y_true[:, :, "target"].sort_index(axis=1)

        intervals_df: pd.DataFrame = y_pred.get_prediction_intervals()
        df_pred_lower = intervals_df.loc[:, pd.IndexSlice[:, lower_border]].sort_index(axis=1)
        df_pred_upper = intervals_df.loc[:, pd.IndexSlice[:, upper_border]].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        values = np.mean(np.abs(df_pred_upper.values - df_pred_lower.values), axis=0)
        metrics_per_segment = dict(zip(segments, values))

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


__all__ = ["Coverage", "Width"]
