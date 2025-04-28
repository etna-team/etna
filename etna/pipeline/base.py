import math
import reprlib
import warnings
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy.stats import norm
from typing_extensions import Self
from typing_extensions import TypedDict
from typing_extensions import assert_never

from etna.core import AbstractSaveable
from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.datasets.utils import _check_timestamp_param
from etna.datasets.utils import timestamp_range
from etna.distributions import BaseDistribution
from etna.loggers import _Logger
from etna.loggers import tslogger
from etna.metrics import BaseMetric
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode
from etna.metrics.functional_metrics import ArrayLike


class CrossValidationMode(str, Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} modes allowed"
        )


class FoldMask(BaseMixin):
    """Container to hold the description of the fold mask.

    Fold masks are expected to be used for backtest strategy customization.
    """

    def __init__(
        self,
        first_train_timestamp: Union[pd.Timestamp, int, str, None],
        last_train_timestamp: Union[pd.Timestamp, int, str],
        target_timestamps: List[Union[pd.Timestamp, int, str]],
    ):
        """Init FoldMask.

        Values of ``target_timestamps`` are sorted in ascending order.

        Notes
        -----
        String value is converted into :py:class`pd.Timestamps` using :py:func:`pandas.to_datetime`.

        Parameters
        ----------
        first_train_timestamp:
            First train timestamp, the first timestamp in the dataset if None is passed
        last_train_timestamp:
            Last train timestamp
        target_timestamps:
            List of target timestamps

        Raises
        ------
        ValueError:
            All timestamps should be one of two possible types: pd.Timestamp or int
        ValueError:
            Last train timestamp should be not sooner than first train timestamp
        ValueError:
            Target timestamps shouldn't be empty
        ValueError:
            Target timestamps shouldn't contain duplicates
        ValueError:
            Target timestamps should be strictly later then last train timestamp
        """
        if isinstance(first_train_timestamp, str):
            first_train_timestamp = pd.to_datetime(first_train_timestamp)

        if isinstance(last_train_timestamp, str):
            last_train_timestamp = pd.to_datetime(last_train_timestamp)

        target_timestamps_processed = []
        for timestamp in target_timestamps:
            if isinstance(timestamp, str):
                target_timestamps_processed.append(pd.to_datetime(timestamp))
            else:
                target_timestamps_processed.append(timestamp)

        self._validate_parameters_same_type(
            first_train_timestamp=first_train_timestamp,
            last_train_timestamp=last_train_timestamp,
            target_timestamps=target_timestamps_processed,
        )

        target_timestamps = sorted(target_timestamps_processed)

        self._validate_first_last_train_timestamps_order(
            first_train_timestamp=first_train_timestamp, last_train_timestamp=last_train_timestamp
        )
        self._validate_target_timestamps(last_train_timestamp=last_train_timestamp, target_timestamps=target_timestamps)

        self.first_train_timestamp = first_train_timestamp
        self.last_train_timestamp = last_train_timestamp
        self.target_timestamps = target_timestamps

    @staticmethod
    def _validate_parameters_same_type(
        first_train_timestamp: Union[pd.Timestamp, int, str, None],
        last_train_timestamp: Union[pd.Timestamp, int],
        target_timestamps: List[Union[pd.Timestamp, int]],
    ):
        """Check that first train timestamp, last train timestamp, target timestamps has the same type."""
        if first_train_timestamp is not None:
            values_to_check = [first_train_timestamp, last_train_timestamp, *target_timestamps]
        else:
            values_to_check = [last_train_timestamp, *target_timestamps]

        types: Set[type] = set()
        for value in values_to_check:
            if isinstance(value, np.integer):
                types.add(int)
            else:
                types.add(type(value))

        if len(types) > 1:
            raise ValueError("All timestamps should be one of two possible types: pd.Timestamp or int!")

    @staticmethod
    def _validate_first_last_train_timestamps_order(
        first_train_timestamp: Union[pd.Timestamp, int, None], last_train_timestamp: Union[pd.Timestamp, int]
    ):
        """Check that last train timestamp is later than first train timestamp."""
        if first_train_timestamp is not None and last_train_timestamp < first_train_timestamp:  # type: ignore
            raise ValueError("Last train timestamp should be not sooner than first train timestamp!")

    @staticmethod
    def _validate_target_timestamps(
        last_train_timestamp: Union[pd.Timestamp, int], target_timestamps: List[Union[pd.Timestamp, int]]
    ):
        """Check that all target timestamps aren't empty and later than last train timestamp."""
        if len(target_timestamps) == 0:
            raise ValueError("Target timestamps shouldn't be empty!")

        if len(target_timestamps) != len(set(target_timestamps)):
            raise ValueError("Target timestamps shouldn't contain duplicates!")

        first_target_timestamp = target_timestamps[0]
        if first_target_timestamp <= last_train_timestamp:  # type: ignore
            raise ValueError("Target timestamps should be strictly later then last train timestamp!")

    def validate_on_dataset(self, ts: TSDataset, horizon: int):
        """Validate fold mask on the dataset with specified horizon.

        Parameters
        ----------
        ts:
            Dataset to validate on
        horizon:
            Forecasting horizon

        Raises
        ------
        ValueError:
            First train timestamp isn't present in a given dataset
        ValueError:
            Last train timestamp isn't present in a given dataset
        ValueError:
            Some of target timestamps aren't present in a given dataset
        ValueError:
            First train timestamp should be later than minimal dataset timestamp
        ValueError:
            Last target timestamp should be not later than horizon steps after last train timestamp
        """
        timestamps = ts.timestamps.to_list()

        if self.first_train_timestamp is not None and self.first_train_timestamp not in timestamps:
            raise ValueError("First train timestamp isn't present in a given dataset!")

        if self.last_train_timestamp not in timestamps:
            raise ValueError("Last train timestamp isn't present in a given dataset!")

        if not set(self.target_timestamps).issubset(set(timestamps)):
            diff = set(self.target_timestamps).difference(set(timestamps))
            raise ValueError(f"Some target timestamps aren't present in a given dataset: {reprlib.repr(diff)}")

        dataset_horizon_border_timestamp = timestamps[timestamps.index(self.last_train_timestamp) + horizon]
        mask_last_target_timestamp = self.target_timestamps[-1]
        if dataset_horizon_border_timestamp < mask_last_target_timestamp:
            raise ValueError(f"Last target timestamp should be not later than {dataset_horizon_border_timestamp}!")


class FoldParallelGroup(TypedDict):
    """Group for parallel fold processing."""

    train_fold_number: int
    train_mask: FoldMask
    forecast_fold_numbers: List[int]
    forecast_masks: List[FoldMask]


class _DummyMetric(Metric):
    """Dummy metric that is created only for implementation of BasePipeline._forecast_prediction_interval."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        super().__init__(mode=mode, metric_fn=self._compute_metric, **kwargs)

    @staticmethod
    def _compute_metric(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        return 0.0

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[Optional[float], Dict[str, Optional[float]]]:
        segments = set(y_true.segments)
        metrics_per_segment: Dict[str, Optional[float]] = {}
        for segment in segments:
            metrics_per_segment[segment] = 0.0
        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics


class BasePipeline(BaseMixin, AbstractSaveable):
    """Base class for pipeline."""

    def __init__(self, horizon: int):
        """
        Create instance of BasePipeline with given parameters.

        Parameters
        ----------
        horizon:
            Number of timestamps in the future for forecasting
        """
        self._validate_horizon(horizon=horizon)
        self.horizon = horizon
        self.ts: Optional[TSDataset] = None

    @abstractmethod
    def fit(self, ts: TSDataset, save_ts: bool = True) -> "BasePipeline":
        """Fit the Pipeline.

        Parameters
        ----------
        ts:
            Dataset with timeseries data
        save_ts:
            Will ``ts`` be saved in the pipeline during ``fit``.

        Returns
        -------
        :
            Fitted Pipeline instance
        """
        pass

    @staticmethod
    def _validate_horizon(horizon: int):
        """Check that given number of folds is grater than 1."""
        if horizon <= 0:
            raise ValueError("At least one point in the future is expected.")

    @staticmethod
    def _validate_quantiles(quantiles: Sequence[float]) -> Sequence[float]:
        """Check that given number of folds is grater than 1."""
        for quantile in quantiles:
            if not (0 < quantile < 1):
                raise ValueError("Quantile should be a number from (0,1).")
        return quantiles

    @abstractmethod
    def _forecast(self, ts: TSDataset, return_components: bool) -> TSDataset:
        """Make predictions."""
        pass

    def _forecast_prediction_interval(
        self, ts: TSDataset, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Add prediction intervals to the forecasts."""
        forecast_ts_list = self.get_historical_forecasts(ts=ts, n_folds=n_folds)

        self._add_forecast_borders(
            ts=ts, list_backtest_forecasts=forecast_ts_list, quantiles=quantiles, predictions=predictions
        )

        return predictions

    @staticmethod
    def _validate_residuals_for_interval_estimation(backtest_forecasts: pd.DataFrame, residuals: pd.DataFrame):
        len_backtest, num_segments = residuals.shape
        min_timestamp = backtest_forecasts.index.min()
        max_timestamp = backtest_forecasts.index.max()
        non_nan_counts = np.sum(~np.isnan(residuals.values), axis=0)
        if np.any(non_nan_counts < len_backtest):
            warnings.warn(
                f"There are NaNs in target on time span from {min_timestamp} to {max_timestamp}. "
                f"It can obstruct prediction interval estimation on history data."
            )
        if np.any(non_nan_counts < 2):
            raise ValueError(
                f"There aren't enough target values to evaluate prediction intervals on history! "
                f"For each segment there should be at least 2 points with defined value in a "
                f"time span from {min_timestamp} to {max_timestamp}. "
                f"You can try to increase n_folds parameter to make time span bigger."
            )

    def _add_forecast_borders(
        self,
        ts: TSDataset,
        list_backtest_forecasts: List[TSDataset],
        quantiles: Sequence[float],
        predictions: TSDataset,
    ) -> None:
        """Estimate prediction intervals and add to the forecasts."""
        backtest_forecasts = pd.concat([forecast.to_pandas() for forecast in list_backtest_forecasts], axis=0)

        target = ts[backtest_forecasts.index.min() : backtest_forecasts.index.max(), :, "target"]
        if not backtest_forecasts.index.equals(target.index):
            raise ValueError("Historical backtest timestamps must match with the original dataset timestamps!")

        residuals = backtest_forecasts.loc[:, pd.IndexSlice[:, "target"]] - target

        self._validate_residuals_for_interval_estimation(backtest_forecasts=backtest_forecasts, residuals=residuals)
        sigma = np.nanstd(residuals.values, axis=0)

        borders = []
        for quantile in quantiles:
            z_q = norm.ppf(q=quantile)
            border = predictions[:, :, "target"] + sigma * z_q
            border.rename({"target": f"target_{quantile:.4g}"}, inplace=True, axis=1)
            borders.append(border)

        quantiles_df = pd.concat(borders, axis=1)
        predictions.add_prediction_intervals(prediction_intervals_df=quantiles_df)

    def forecast(
        self,
        ts: Optional[TSDataset] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        n_folds: int = 3,
        return_components: bool = False,
    ) -> TSDataset:
        """Make a forecast of the next points of a dataset.

        The result of forecasting starts from the last point of ``ts``, not including it.

        Parameters
        ----------
        ts:
            Dataset to forecast. If not given, dataset given during :py:meth:`fit` is used.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions

        Raises
        ------
        NotImplementedError:
            Adding target components is not currently implemented
        """
        if ts is None:
            if self.ts is None:
                raise ValueError(
                    "There is no ts to forecast! Pass ts into forecast method or make sure that pipeline contains ts."
                )
            ts = self.ts

        self._validate_quantiles(quantiles=quantiles)
        self._validate_backtest_n_folds(n_folds=n_folds)

        predictions = self._forecast(ts=ts, return_components=return_components)
        if prediction_interval:
            predictions = self._forecast_prediction_interval(
                ts=ts, predictions=predictions, quantiles=quantiles, n_folds=n_folds
            )
        return predictions

    @staticmethod
    def _make_predict_timestamps(
        ts: TSDataset,
        start_timestamp: Union[pd.Timestamp, int, str, None],
        end_timestamp: Union[pd.Timestamp, int, str, None],
    ) -> Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[int, int]]:
        start_timestamp = _check_timestamp_param(param=start_timestamp, param_name="start_timestamp", freq=ts.freq)
        end_timestamp = _check_timestamp_param(param=end_timestamp, param_name="end_timestamp", freq=ts.freq)

        min_timestamp = ts.describe()["start_timestamp"].max()
        max_timestamp = ts.timestamps[-1]

        if start_timestamp is None:
            start_timestamp = min_timestamp
        if end_timestamp is None:
            end_timestamp = max_timestamp

        if start_timestamp < min_timestamp:
            raise ValueError("Value of start_timestamp is less than beginning of some segments!")
        if end_timestamp > max_timestamp:
            raise ValueError("Value of end_timestamp is more than ending of dataset!")

        if start_timestamp > end_timestamp:  # type: ignore
            raise ValueError("Value of end_timestamp is less than start_timestamp!")

        return start_timestamp, end_timestamp

    @abstractmethod
    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: Union[pd.Timestamp, int],
        end_timestamp: Union[pd.Timestamp, int],
        prediction_interval: bool,
        quantiles: Sequence[float],
        return_components: bool,
    ) -> TSDataset:
        pass

    def predict(
        self,
        ts: TSDataset,
        start_timestamp: Union[pd.Timestamp, int, str, None] = None,
        end_timestamp: Union[pd.Timestamp, int, str, None] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
    ) -> TSDataset:
        """Make in-sample predictions on dataset in a given range.

        Currently, in situation when segments start with different timestamps
        we only guarantee to work with ``start_timestamp`` >= beginning of all segments.

        Parameters ``start_timestamp`` and ``end_timestamp`` of type ``str`` are converted into ``pd.Timestamp``.

        Parameters
        ----------
        ts:
            Dataset to make predictions on.
        start_timestamp:
            First timestamp of prediction range to return, should be >= than first timestamp in ``ts``;
            expected that beginning of each segment <= ``start_timestamp``;
            if isn't set the first timestamp where each segment began is taken.
        end_timestamp:
            Last timestamp of prediction range to return; if isn't set the last timestamp of ``ts`` is taken.
            Expected that value is less or equal to the last timestamp in ``ts``.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions in ``[start_timestamp, end_timestamp]`` range.

        Raises
        ------
        ValueError
            Incorrect type of ``start_timestamp`` or ``end_timestamp`` is used according to ``ts.freq``
        ValueError:
            Value of ``end_timestamp`` is less than ``start_timestamp``.
        ValueError:
            Value of ``start_timestamp`` goes before point where each segment started.
        ValueError:
            Value of ``end_timestamp`` goes after the last timestamp.
        NotImplementedError:
            Adding target components is not currently implemented
        """
        start_timestamp, end_timestamp = self._make_predict_timestamps(
            ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )
        self._validate_quantiles(quantiles=quantiles)
        result = self._predict(
            ts=ts,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            return_components=return_components,
        )
        return result

    def _init_backtest(self):
        self._folds: Optional[Dict[int, Any]] = None
        self._fold_column = "fold_number"

    @staticmethod
    def _validate_backtest_n_folds(n_folds: int):
        """Check that given n_folds value is >= 1."""
        if n_folds < 1:
            raise ValueError(f"Folds number should be a positive number, {n_folds} given")

    @staticmethod
    def _validate_backtest_mode(n_folds: Union[int, List[FoldMask]], mode: Optional[str]) -> CrossValidationMode:
        if mode is None:
            return CrossValidationMode.expand

        if not isinstance(n_folds, int):
            raise ValueError("Mode shouldn't be set if n_folds are fold masks!")

        return CrossValidationMode(mode.lower())

    @staticmethod
    def _validate_backtest_stride(n_folds: Union[int, List[FoldMask]], horizon: int, stride: Optional[int]) -> int:
        if stride is None:
            return horizon

        if not isinstance(n_folds, int):
            raise ValueError("Stride shouldn't be set if n_folds are fold masks!")

        if stride < 1:
            raise ValueError(f"Stride should be a positive number, {stride} given!")

        return stride

    @staticmethod
    def _validate_backtest_dataset(ts: TSDataset, n_folds: int, horizon: int, stride: int):
        """Check all segments have enough timestamps to validate forecaster with given number of splits."""
        min_required_length = horizon + (n_folds - 1) * stride

        df = ts._df.loc[:, pd.IndexSlice[:, "target"]]
        num_timestamps = df.shape[0]
        not_na = ~np.isnan(df.values)
        min_idx = np.argmax(not_na, axis=0)

        short_history_mask = np.logical_or((num_timestamps - min_idx) < min_required_length, np.all(~not_na, axis=0))
        short_segments = np.array(ts.segments)[short_history_mask]
        if len(short_segments) > 0:
            raise ValueError(
                f"All the series from feature dataframe should contain at least "
                f"{horizon} + {n_folds - 1} * {stride} = {min_required_length} timestamps; "
                f"series {short_segments[0]} does not."
            )

    @staticmethod
    def _generate_masks_from_n_folds(
        ts: TSDataset, n_folds: int, horizon: int, mode: CrossValidationMode, stride: int
    ) -> List[FoldMask]:
        """Generate fold masks from n_folds."""
        if mode is CrossValidationMode.expand:
            constant_history_length = 0
        elif mode is CrossValidationMode.constant:
            constant_history_length = 1
        else:
            assert_never(mode)

        masks = []
        dataset_timestamps = list(ts.timestamps)
        min_timestamp_idx, max_timestamp_idx = 0, len(dataset_timestamps)
        for offset in range(n_folds, 0, -1):
            min_train_idx = min_timestamp_idx + (n_folds - offset) * stride * constant_history_length
            max_train_idx = max_timestamp_idx - stride * (offset - 1) - horizon - 1
            min_test_idx = max_train_idx + 1
            max_test_idx = max_train_idx + horizon

            min_train, max_train = dataset_timestamps[min_train_idx], dataset_timestamps[max_train_idx]
            min_test, max_test = dataset_timestamps[min_test_idx], dataset_timestamps[max_test_idx]
            target_timestamps = timestamp_range(start=min_test, end=max_test, freq=ts.freq).tolist()
            mask = FoldMask(
                first_train_timestamp=min_train,
                last_train_timestamp=max_train,
                target_timestamps=target_timestamps,
            )
            masks.append(mask)

        return masks

    @staticmethod
    def _validate_backtest_metrics(metrics: List[BaseMetric]):
        """Check that given metrics are valid for backtest."""
        if not metrics:
            raise ValueError("At least one metric required")
        for metric in metrics:
            if not metric.mode == MetricAggregationMode.per_segment:
                raise ValueError(
                    f"All the metrics should be in {MetricAggregationMode.per_segment}, "
                    f"{metric.name} metric is in {metric.mode} mode"
                )

    @staticmethod
    def _generate_folds_datasets(
        ts: TSDataset, masks: List[FoldMask], horizon: int
    ) -> Generator[Tuple[TSDataset, TSDataset], None, None]:
        """Generate folds."""
        timestamps = list(ts.timestamps)
        for mask in masks:
            min_train_idx = timestamps.index(mask.first_train_timestamp)
            max_train_idx = timestamps.index(mask.last_train_timestamp)
            min_test_idx = max_train_idx + 1
            max_test_idx = max_train_idx + horizon

            min_train, max_train = timestamps[min_train_idx], timestamps[max_train_idx]
            min_test, max_test = timestamps[min_test_idx], timestamps[max_test_idx]

            train, test = ts.train_test_split(
                train_start=min_train, train_end=max_train, test_start=min_test, test_end=max_test
            )
            yield train, test

    def _compute_metrics(
        self, metrics: List[BaseMetric], y_true: TSDataset, y_pred: TSDataset
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for given y_true, y_pred."""
        if y_true.has_hierarchy():
            if y_true.current_df_level != y_pred.current_df_level:
                y_true = y_true.get_level_dataset(y_pred.current_df_level)  # type: ignore

        metrics_values: Dict[str, Dict[str, float]] = {}
        for metric in metrics:
            metrics_values[metric.name] = metric(y_true=y_true, y_pred=y_pred)  # type: ignore
        return metrics_values

    def _fit_backtest_pipeline(self, ts: TSDataset, fold_number: int, logger: _Logger) -> "BasePipeline":
        """Fit pipeline for a given data in backtest."""
        with logger.capture_tslogger():
            logger.start_experiment(job_type="training", group=str(fold_number))
            pipeline = deepcopy(self)
            pipeline.fit(ts=ts, save_ts=False)
            logger.finish_experiment()
        return pipeline

    def _forecast_backtest_pipeline(
        self,
        pipeline: "BasePipeline",
        ts: TSDataset,
        fold_number: int,
        forecast_params: Dict[str, Any],
        logger: _Logger,
    ) -> TSDataset:
        """Make a forecast with a given pipeline in backtest."""
        with logger.capture_tslogger():
            logger.start_experiment(job_type="forecasting", group=str(fold_number))
            forecast = pipeline.forecast(ts=ts, **forecast_params)
            logger.finish_experiment()
        return forecast

    def _process_fold_forecast(
        self,
        forecast: TSDataset,
        train: TSDataset,
        test: TSDataset,
        pipeline: "BasePipeline",
        fold_number: int,
        mask: FoldMask,
        metrics: List[BaseMetric],
        logger: _Logger,
    ) -> Dict[str, Any]:
        """Process forecast made for a fold."""
        with logger.capture_tslogger():
            logger.start_experiment(job_type="crossval", group=str(fold_number))

            fold: Dict[str, Any] = {}
            for stage_name, stage_ts in zip(("train", "test"), (train, test)):
                fold[f"{stage_name}_timerange"] = {}
                fold[f"{stage_name}_timerange"]["start"] = stage_ts.timestamps.min()
                fold[f"{stage_name}_timerange"]["end"] = stage_ts.timestamps.max()

            forecast._df = forecast._df.loc[mask.target_timestamps]
            test._df = test._df.loc[mask.target_timestamps]

            fold["forecast"] = forecast
            fold["metrics"] = deepcopy(pipeline._compute_metrics(metrics=metrics, y_true=test, y_pred=forecast))

            logger.log_backtest_run(pd.DataFrame(fold["metrics"]), forecast.to_pandas(), test.to_pandas())
            logger.finish_experiment()

        return fold

    def _get_backtest_metrics(self, aggregate_metrics: bool = False) -> pd.DataFrame:
        """Get dataframe with metrics."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        metrics_dfs = []

        for i, fold in self._folds.items():
            fold_metrics = pd.DataFrame(fold["metrics"]).reset_index().rename({"index": "segment"}, axis=1)
            fold_metrics[self._fold_column] = i
            metrics_dfs.append(fold_metrics)
        metrics_df = pd.concat(metrics_dfs)
        metrics_df.sort_values(["segment", self._fold_column], inplace=True)

        if aggregate_metrics:
            metrics_df = (
                metrics_df.groupby("segment").mean(numeric_only=False).reset_index().drop(self._fold_column, axis=1)
            )

        return metrics_df

    def _get_fold_info(self) -> pd.DataFrame:
        """Get information about folds."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        timerange_dfs = []
        for fold_number, fold_info in self._folds.items():
            tmp_df = pd.DataFrame()
            for stage_name in ("train", "test"):
                for border in ("start", "end"):
                    tmp_df[f"{stage_name}_{border}_time"] = [fold_info[f"{stage_name}_timerange"][border]]
            tmp_df[self._fold_column] = fold_number
            timerange_dfs.append(tmp_df)
        timerange_df = pd.concat(timerange_dfs, ignore_index=True)
        return timerange_df

    def _prepare_fold_masks(
        self, ts: TSDataset, masks: Union[int, List[FoldMask]], mode: CrossValidationMode, stride: int
    ) -> List[FoldMask]:
        """Prepare and validate fold masks."""
        if isinstance(masks, int):
            self._validate_backtest_n_folds(n_folds=masks)
            self._validate_backtest_dataset(ts=ts, n_folds=masks, horizon=self.horizon, stride=stride)
            masks = self._generate_masks_from_n_folds(
                ts=ts, n_folds=masks, horizon=self.horizon, mode=mode, stride=stride
            )
        for i, mask in enumerate(masks):
            mask.first_train_timestamp = mask.first_train_timestamp if mask.first_train_timestamp else ts.timestamps[0]
            masks[i] = mask
        for mask in masks:
            mask.validate_on_dataset(ts=ts, horizon=self.horizon)
        return masks

    @staticmethod
    def _make_backtest_fold_groups(masks: List[FoldMask], refit: Union[bool, int]) -> List[FoldParallelGroup]:
        """Make groups of folds for backtest."""
        if not refit:
            refit = len(masks)

        grouped_folds = []
        num_groups = math.ceil(len(masks) / refit)
        for group_id in range(num_groups):
            train_fold_number = group_id * refit
            forecast_fold_numbers = [train_fold_number + i for i in range(refit) if train_fold_number + i < len(masks)]
            cur_group: FoldParallelGroup = {
                "train_fold_number": train_fold_number,
                "train_mask": masks[train_fold_number],
                "forecast_fold_numbers": forecast_fold_numbers,
                "forecast_masks": [masks[i] for i in forecast_fold_numbers],
            }
            grouped_folds.append(cur_group)

        return grouped_folds

    def _run_all_folds(
        self,
        masks: List[FoldMask],
        ts: TSDataset,
        metrics: List[BaseMetric],
        n_jobs: int,
        refit: Union[bool, int],
        joblib_params: Dict[str, Any],
        forecast_params: Dict[str, Any],
    ) -> Tuple[Dict[int, Any], List["BasePipeline"]]:
        """Run pipeline on all folds."""
        fold_groups = self._make_backtest_fold_groups(masks=masks, refit=refit)

        with Parallel(n_jobs=n_jobs, **joblib_params) as parallel:
            # fitting
            fit_masks = [group["train_mask"] for group in fold_groups]
            fit_datasets = (
                train for train, _ in self._generate_folds_datasets(ts=ts, masks=fit_masks, horizon=self.horizon)
            )
            pipelines = parallel(
                delayed(self._fit_backtest_pipeline)(
                    ts=fit_ts, fold_number=fold_groups[group_idx]["train_fold_number"], logger=tslogger
                )
                for group_idx, fit_ts in enumerate(fit_datasets)
            )

            # forecasting
            forecast_masks = [group["forecast_masks"] for group in fold_groups]
            forecast_datasets = (
                (
                    train
                    for train, _ in self._generate_folds_datasets(
                        ts=ts, masks=group_forecast_masks, horizon=self.horizon
                    )
                )
                for group_forecast_masks in forecast_masks
            )
            forecasts_flat = parallel(
                delayed(self._forecast_backtest_pipeline)(
                    ts=forecast_ts,
                    pipeline=pipelines[group_idx],
                    fold_number=fold_groups[group_idx]["forecast_fold_numbers"][idx],
                    forecast_params=forecast_params,
                    logger=tslogger,
                )
                for group_idx, group_forecast_datasets in enumerate(forecast_datasets)
                for idx, forecast_ts in enumerate(group_forecast_datasets)
            )

            # processing forecasts
            fold_process_train_datasets = (
                train for train, _ in self._generate_folds_datasets(ts=ts, masks=fit_masks, horizon=self.horizon)
            )
            fold_process_test_datasets = (
                (
                    test
                    for _, test in self._generate_folds_datasets(
                        ts=ts, masks=group_forecast_masks, horizon=self.horizon
                    )
                )
                for group_forecast_masks in forecast_masks
            )
            fold_results_flat = parallel(
                delayed(self._process_fold_forecast)(
                    forecast=forecasts_flat[group_idx * refit + idx],
                    train=train,
                    test=test,
                    pipeline=pipelines[group_idx],
                    fold_number=fold_groups[group_idx]["forecast_fold_numbers"][idx],
                    mask=fold_groups[group_idx]["forecast_masks"][idx],
                    metrics=metrics,
                    logger=tslogger,
                )
                for group_idx, (train, group_fold_process_test_datasets) in enumerate(
                    zip(fold_process_train_datasets, fold_process_test_datasets)
                )
                for idx, test in enumerate(group_fold_process_test_datasets)
            )

        results = {
            fold_number: fold_results_flat[group_idx * refit + idx]
            for group_idx in range(len(fold_groups))
            for idx, fold_number in enumerate(fold_groups[group_idx]["forecast_fold_numbers"])
        }
        return results, pipelines

    def backtest(
        self,
        ts: TSDataset,
        metrics: List[BaseMetric],
        n_folds: Union[int, List[FoldMask]] = 5,
        mode: Optional[str] = None,
        aggregate_metrics: bool = False,
        n_jobs: int = 1,
        refit: Union[bool, int] = True,
        stride: Optional[int] = None,
        joblib_params: Optional[Dict[str, Any]] = None,
        forecast_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[pd.DataFrame, List[TSDataset], List[Self]]]:
        """Run backtest with the pipeline.

        If ``refit != True`` and some component of the pipeline doesn't support forecasting with gap, this component will raise an exception.

        Parameters
        ----------
        ts:
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds or the list of fold masks
        mode:
            Train generation policy: 'expand' or 'constant'. Works only if ``n_folds`` is integer.
            By default, is set to 'expand'.
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        refit:
            Determines how often pipeline should be retrained during iteration over folds.

            * If ``True``: pipeline is retrained on each fold.

            * If ``False``: pipeline is trained only on the first fold.

            * If ``value: int``: pipeline is trained every ``value`` folds starting from the first.

        stride:
            Number of points between folds. Works only if ``n_folds`` is integer. By default, is set to ``horizon``.
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`
        forecast_params:
            Additional parameters for :py:func:`~etna.pipeline.base.BasePipeline.forecast`

        Returns
        -------
        backtest_result:
            Dictionary with backtest results. It contains metrics dataframe, list of TSDatasets with forecast for each fold in ascending order folds,
            dataframe with information about folds and list of pipelines for each fold in ascending order folds.


        Raises
        ------
        ValueError:
            If ``mode`` is set when ``n_folds`` are ``List[FoldMask]``.
        ValueError:
            If ``stride`` is set when ``n_folds`` are ``List[FoldMask]``.
        """
        mode_enum = self._validate_backtest_mode(n_folds=n_folds, mode=mode)
        stride = self._validate_backtest_stride(n_folds=n_folds, horizon=self.horizon, stride=stride)

        if joblib_params is None:
            joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")

        if forecast_params is None:
            forecast_params = dict()

        self._init_backtest()
        self._validate_backtest_metrics(metrics=metrics)
        masks = self._prepare_fold_masks(ts=ts, masks=n_folds, mode=mode_enum, stride=stride)
        self._folds, pipelines = self._run_all_folds(
            masks=masks,
            ts=ts,
            metrics=metrics,
            n_jobs=n_jobs,
            refit=refit,
            joblib_params=joblib_params,
            forecast_params=forecast_params,
        )

        metrics_df = self._get_backtest_metrics(aggregate_metrics=aggregate_metrics)
        forecast_ts_list = [fold["forecast"] for fold in self._folds.values()]
        fold_info_df = self._get_fold_info()

        tslogger.start_experiment(job_type="crossval_results", group="all")
        tslogger.log_backtest_metrics(ts, metrics_df, forecast_ts_list, fold_info_df)
        tslogger.finish_experiment()

        backtest_result = {
            "metrics": metrics_df,
            "forecasts": forecast_ts_list,
            "fold_info": fold_info_df,
            "pipelines": pipelines,
        }

        return backtest_result

    def get_historical_forecasts(
        self,
        ts: TSDataset,
        n_folds: Union[int, List[FoldMask]] = 5,
        mode: Optional[str] = None,
        n_jobs: int = 1,
        refit: Union[bool, int] = True,
        stride: Optional[int] = None,
        joblib_params: Optional[Dict[str, Any]] = None,
        forecast_params: Optional[Dict[str, Any]] = None,
    ) -> List[TSDataset]:
        """Estimate forecast for each fold on the historical dataset.

        If ``refit != True`` and some component of the pipeline doesn't support forecasting with gap, this component will raise an exception.

        Parameters
        ----------
        ts:
            Dataset to fit models in backtest
        n_folds:
            Number of folds or the list of fold masks
        mode:
            Train generation policy: 'expand' or 'constant'. Works only if ``n_folds`` is integer.
            By default, is set to 'expand'.
        n_jobs:
            Number of jobs to run in parallel
        refit:
            Determines how often pipeline should be retrained during iteration over folds.

            * If ``True``: pipeline is retrained on each fold.

            * If ``False``: pipeline is trained only on the first fold.

            * If ``value: int``: pipeline is trained every ``value`` folds starting from the first.

        stride:
            Number of points between folds. Works only if ``n_folds`` is integer. By default, is set to ``horizon``.
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`
        forecast_params:
            Additional parameters for :py:func:`~etna.pipeline.base.BasePipeline.forecast`

        Returns
        -------
        :
            List of `TSDataset` with forecast for each fold on the historical dataset.

        Raises
        ------
        ValueError:
            If ``mode`` is set when ``n_folds`` are ``List[FoldMask]``.
        ValueError:
            If ``stride`` is set when ``n_folds`` are ``List[FoldMask]``.
        """
        with tslogger.disable():
            backtest_result = self.backtest(
                ts=ts,
                metrics=[_DummyMetric()],
                n_folds=n_folds,
                mode=mode,
                n_jobs=n_jobs,
                refit=refit,
                stride=stride,
                joblib_params=joblib_params,
                forecast_params=forecast_params,
            )
            forecast_ts_list = backtest_result["forecasts"]
            forecast_ts_list = cast(List[TSDataset], forecast_ts_list)

        return forecast_ts_list

    @abstractmethod
    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get hyperparameter grid to tune.

        Returns
        -------
        :
            Grid with hyperparameters.
        """
        pass
