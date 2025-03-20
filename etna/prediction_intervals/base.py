from abc import abstractmethod
from typing import Dict
from typing import Optional
from typing import Sequence

import pandas as pd

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.pipeline.base import BasePipeline
from etna.prediction_intervals.mixins import SavePredictionIntervalsMixin


class BasePredictionIntervals(SavePredictionIntervalsMixin, BasePipeline):
    """Base class for prediction intervals methods.

    This class implements a wrapper interface for pipelines and ensembles that provides the ability to
    estimate prediction intervals.

    To implement a particular method, one must inherit from this class and provide an implementation for the
    abstract method ``_forecast_prediction_interval``. This method should estimate and store prediction
    intervals for out-of-sample forecasts.

    In-sample prediction is not supported by default and will raise a corresponding error while attempting to do so.
    This functionality could be implemented if needed by overriding ``_predict`` method. This method is responsible
    for building an in-sample point forecast and adding prediction intervals.
    """

    def __init__(self, pipeline: BasePipeline):
        """Initialize instance of ``BasePredictionIntervals`` with given parameters.

        Parameters
        ----------
        pipeline:
            Base pipeline or ensemble for prediction intervals estimation.
        """
        ts = pipeline.ts
        self.pipeline = pipeline
        super().__init__(pipeline.horizon)
        self.pipeline.ts = ts

    def fit(self, ts: TSDataset, save_ts: bool = True) -> "BasePredictionIntervals":
        """Fit the pipeline or ensemble of pipelines.

        Fit and apply given transforms to the data, then fit the model on the transformed data.

        Parameters
        ----------
        ts:
            Dataset with timeseries data.
        save_ts:
            Whether to save ``ts`` in the pipeline during ``fit``.

        Returns
        -------
        :
            Fitted instance.
        """
        self.pipeline.fit(ts=ts, save_ts=save_ts)
        return self

    @property
    def ts(self) -> Optional[TSDataset]:
        """Access internal pipeline dataset."""
        return self.pipeline.ts

    @ts.setter
    def ts(self, ts: Optional[TSDataset]):
        """Set internal pipeline dataset."""
        self.pipeline.ts = ts

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp],
        end_timestamp: Optional[pd.Timestamp],
        prediction_interval: bool,
        quantiles: Sequence[float],
        return_components: bool,
    ) -> TSDataset:
        """Make in-sample predictions on dataset in a given range.

        This method is not implemented by default. A custom implementation could be added by overriding if needed.

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
            If ``True`` returns prediction interval.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
        return_components:
            If ``True`` additionally returns forecast components.

        Returns
        -------
        :
            Dataset with predictions in ``[start_timestamp, end_timestamp]`` range.
        """
        raise NotImplementedError(
            "In-sample sample prediction is not supported! See documentation on how it could be implemented."
        )

    def _forecast(self, ts: TSDataset, return_components: bool) -> TSDataset:
        """Make point forecasts using base pipeline or ensemble."""
        return self.pipeline._forecast(ts=ts, return_components=return_components)

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
            Dataset to forecast.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
            If method don't use or estimate quantiles this parameter will be ignored.
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation.
        return_components:
            If True additionally returns forecast components.

        Returns
        -------
        :
            Dataset with predictions.
        """
        predictions = super().forecast(
            ts=ts,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            n_folds=n_folds,
            return_components=return_components,
        )
        return predictions

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get hyperparameter grid of the base pipeline to tune.

        Returns
        -------
        :
            Grid with hyperparameters.
        """
        pipeline_params = self.pipeline.params_to_tune()
        pipeline_params = {f"pipeline.{key}": value for key, value in pipeline_params.items()}
        return pipeline_params

    @abstractmethod
    def _forecast_prediction_interval(
        self, ts: TSDataset, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Estimate and store prediction intervals.

        Parameters
        ----------
        ts:
            Dataset to forecast.
        predictions:
            Dataset with point predictions.
        quantiles:
            Levels of prediction distribution.
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation.

        Returns
        -------
        :
            Dataset with predictions.
        """
        pass
