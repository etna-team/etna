import reprlib
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from typing_extensions import Protocol
from typing_extensions import assert_never

from etna.core import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.metrics.functional_metrics import ArrayLike


class MetricAggregationMode(str, Enum):
    """Enum for different metric aggregation modes."""

    #: Metric is calculated for each segment and averaged.
    macro = "macro"

    #: Metric is calculated for each segment.
    per_segment = "per-segment"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregation allowed"
        )


class MetricMissingMode(str, Enum):
    """Enum for different metric modes of working with missing values."""

    #: The error is raised on missing values in y_true or y_pred.
    error = "error"

    #: Missing values in y_true are ignored, the error is raised on missing values in y_pred.
    ignore = "ignore"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} modes allowed"
        )


class MetricFunctionSignature(str, Enum):
    """Enum for different metric function signatures."""

    #: function should expect arrays of y_pred and y_true with length ``n_timestamps`` and return scalar
    array_to_scalar = "array_to_scalar"

    #: function should expect matrices of y_pred and y_true with shape ``(n_timestamps, n_segments)``
    #: and return vector of length ``n_segments``
    matrix_to_array = "matrix_to_array"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} signatures allowed"
        )


class MetricFunction(Protocol):
    """Protocol for ``metric_fn`` parameter."""

    @abstractmethod
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        pass


class BaseMetric(BaseMixin):
    """Base class for metric."""

    def __init__(self, mode: str = "per-segment"):
        """
        Init Metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        Raises
        ------
        NotImplementedError:
            If non-existent ``mode`` is used.
        """
        self._aggregate_metrics: Callable[
            [Dict[str, Optional[float]]], Union[Optional[float], Dict[str, Optional[float]]]
        ]
        mode_enum = MetricAggregationMode(mode)
        if mode_enum is MetricAggregationMode.macro:
            self._aggregate_metrics = self._macro_average
        elif mode_enum is MetricAggregationMode.per_segment:
            self._aggregate_metrics = self._per_segment_average
        else:
            assert_never(mode_enum)

        self.mode = mode

    @staticmethod
    def _macro_average(metrics_per_segments: Dict[str, Optional[float]]) -> Optional[float]:
        """
        Compute macro averaging of metrics over segment.

        Parameters
        ----------
        metrics_per_segments:
            dict of {segment: metric_value} for segments to aggregate

        Returns
        -------
        :
            aggregated value of metric
        """
        return np.mean(list(metrics_per_segments.values())).item()  # type: ignore

    @staticmethod
    def _per_segment_average(metrics_per_segments: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Compute per-segment averaging of metrics over segment.

        Parameters
        ----------
        metrics_per_segments:
            dict of {segment: metric_value} for segments to aggregate

        Returns
        -------
        :
            aggregated dict of metric
        """
        return metrics_per_segments

    def _log_start(self):
        """Log metric computation."""
        tslogger.log(f"Metric {self.__repr__()} is calculated on dataset")

    @abstractmethod
    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[Optional[float], Dict[str, Optional[float]]]:
        """
        Compute metric's value with ``y_true`` and ``y_pred``.

        Notes
        -----
        Note that if ``y_true`` and ``y_pred`` are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
        :
            metric's value aggregated over segments or not (depends on mode)
        """
        pass

    def _validate_base(self, y_true: TSDataset, y_pred: TSDataset):
        """
        Check that ``y_true`` and ``y_pred`` pass all base validations.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset
        """
        self._validate_segments(y_true=y_true, y_pred=y_pred)
        self._validate_target_columns(y_true=y_true, y_pred=y_pred)
        self._validate_index(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _validate_segments(y_true: TSDataset, y_pred: TSDataset):
        """Check that segments in ``y_true`` and ``y_pred`` are the same.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            if there are mismatches in y_true and y_pred segments
        """
        segments_true = set(y_true.segments)
        segments_pred = set(y_pred.segments)

        pred_diff_true = segments_pred - segments_true
        true_diff_pred = segments_true - segments_pred
        if pred_diff_true:
            raise ValueError(
                f"There are segments in y_pred that are not in y_true, for example: "
                f"{', '.join(list(pred_diff_true)[:5])}"
            )
        if true_diff_pred:
            raise ValueError(
                f"There are segments in y_true that are not in y_pred, for example: "
                f"{', '.join(list(true_diff_pred)[:5])}"
            )

    @staticmethod
    def _validate_target_columns(y_true: TSDataset, y_pred: TSDataset):
        """Check that ``y_true`` and ``y_pred`` has 'target' feature.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            if y_true or y_pred doesn't contain 'target' feature.
        """
        for name, dataset in zip(("y_true", "y_pred"), (y_true, y_pred)):
            if "target" not in dataset.features:
                raise ValueError(f"{name} should contain 'target' feature.")

    @staticmethod
    def _validate_index(y_true: TSDataset, y_pred: TSDataset):
        """Check that ``y_true`` and ``y_pred`` have the same timestamps.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            If there are mismatches in ``y_true`` and ``y_pred`` timestamps
        """
        if not y_true.timestamps.equals(y_pred.timestamps):
            raise ValueError("y_true and y_pred have different timestamps")

    @property
    def name(self) -> str:
        """Name of the metric for representation."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def greater_is_better(self) -> Optional[bool]:
        """Whether higher metric value is better."""
        pass


class Metric(BaseMetric):
    """
    Base class for all the multi-segment metrics.

    How it works: Metric computes ``metric_fn`` value for each segment in given forecast
    dataset and aggregates it according to mode.
    """

    def __init__(
        self,
        metric_fn: MetricFunction,
        mode: str = "per-segment",
        metric_fn_signature: str = "array_to_scalar",
        **kwargs,
    ):
        """
        Init Metric.

        Parameters
        ----------
        metric_fn:
            functional metric
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        metric_fn_signature:
            type of signature of ``metric_fn`` (see :py:class:`~etna.metrics.base.MetricFunctionSignature`)
        kwargs:
            functional metric's params

        Raises
        ------
        NotImplementedError:
            If non-existent ``mode`` is used.
        NotImplementedError:
            If non-existent ``metric_fn_signature`` is used.
        """
        super().__init__(mode)

        self._metric_fn_signature = MetricFunctionSignature(metric_fn_signature)

        self.metric_fn = metric_fn
        self.kwargs = kwargs
        self.metric_fn_signature = metric_fn_signature

    def _validate_nans(self, y_true: TSDataset, y_pred: TSDataset):
        """Check that ``y_true`` and ``y_pred`` doesn't have NaNs.

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
        if (df_true_isna_sum > 0).any():
            error_segments = set(df_true_isna_sum[df_true_isna_sum > 0].index.droplevel("feature").tolist())
            raise ValueError(f"There are NaNs in y_true! Segments with NaNs: {reprlib.repr(error_segments)}.")

        df_pred_isna_sum = df_pred.isna().sum()
        if (df_pred_isna_sum > 0).any():
            error_segments = set(df_pred_isna_sum[df_pred_isna_sum > 0].index.droplevel("feature").tolist())
            raise ValueError(f"There are NaNs in y_pred Segments with NaNs: {reprlib.repr(error_segments)}.")

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[Optional[float], Dict[str, Optional[float]]]:
        """
        Compute metric's value with ``y_true`` and ``y_pred``.

        Notes
        -----
        Note that if ``y_true`` and ``y_pred`` are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
        :
            metric's value aggregated over segments or not (depends on mode)
        """
        self._log_start()
        self._validate_base(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)

        df_true = y_true[:, :, "target"].sort_index(axis=1)
        df_pred = y_pred[:, :, "target"].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        metrics_per_segment: Dict[str, Optional[float]]
        if self._metric_fn_signature is MetricFunctionSignature.array_to_scalar:
            metrics_per_segment = {}
            for i, cur_segment in enumerate(segments):
                cur_y_true = df_true.iloc[:, i].values
                cur_y_pred = df_pred.iloc[:, i].values
                cur_value = self.metric_fn(y_true=cur_y_true, y_pred=cur_y_pred, **self.kwargs)
                cur_value = cast(float, cur_value)
                if np.isnan(cur_value):
                    metrics_per_segment[cur_segment] = None
                else:
                    metrics_per_segment[cur_segment] = cur_value
        elif self._metric_fn_signature is MetricFunctionSignature.matrix_to_array:
            values = self.metric_fn(y_true=df_true.values, y_pred=df_pred.values, **self.kwargs)
            values = cast(Sequence[float], values)
            metrics_per_segment = {}
            for cur_segment, cur_value in zip(segments, values):
                if np.isnan(cur_value):
                    metrics_per_segment[cur_segment] = None
                else:
                    metrics_per_segment[cur_segment] = cur_value
        else:
            assert_never(self._metric_fn_signature)

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics


class MetricWithMissingHandling(Metric):
    """Base class for all the multi-segment metrics that can handle missing values."""

    def __init__(
        self,
        metric_fn: MetricFunction,
        mode: str = "per-segment",
        metric_fn_signature: str = "array_to_scalar",
        missing_mode: str = "error",
        **kwargs,
    ):
        """
        Init Metric.

        Parameters
        ----------
        metric_fn:
            functional metric
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        metric_fn_signature:
            type of signature of ``metric_fn`` (see :py:class:`~etna.metrics.base.MetricFunctionSignature`)
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            functional metric's params

        Raises
        ------
        NotImplementedError:
            If non-existent ``mode`` is used.
        NotImplementedError:
            If non-existent ``metric_fn_signature`` is used.
        NotImplementedError:
            If non-existent ``missing_mode`` is used.
        """
        super().__init__(metric_fn=metric_fn, mode=mode, metric_fn_signature=metric_fn_signature, **kwargs)
        self.missing_mode = missing_mode
        self._missing_mode_enum = MetricMissingMode(missing_mode)

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
            raise ValueError(f"There are NaNs in y_true! Segments with NaNs: {reprlib.repr(error_segments)}.")

        df_pred_isna_sum = df_pred.isna().sum()
        if (df_pred_isna_sum > 0).any():
            error_segments = set(df_pred_isna_sum[df_pred_isna_sum > 0].index.droplevel("feature").tolist())
            raise ValueError(f"There are NaNs in y_pred Segments with NaNs: {reprlib.repr(error_segments)}.")

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


__all__ = ["Metric", "MetricWithMissingHandling", "MetricAggregationMode", "MetricMissingMode", "BaseMetric"]
