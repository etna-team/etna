from typing import Sequence

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.pipeline import BasePipeline
from etna.prediction_intervals import BasePredictionIntervals
from etna.prediction_intervals.utils import residuals_matrices


class EmpiricalPredictionIntervals(BasePredictionIntervals):
    """Estimate prediction intervals using values of historical residuals.

    1. Compute matrix of residuals  :math:`r_{it} = |\hat y_{it} - y_{it}|` using k-fold backtest, where :math:`i` is fold index.

    2. Estimate quantiles levels, that satisfy the provided coverage, for the corresponding residuals distributions.

    3. Estimate quantiles for each timestamp using computed residuals and levels.

    `Reference implementation <https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.conformal.ConformalIntervals.html>`_.
    """

    def __init__(self, pipeline: BasePipeline, coverage: float = 0.95, include_forecast: bool = True, stride: int = 1):
        """Initialize instance of ``EmpiricalPredictionIntervals`` with given parameters.

        Parameters
        ----------
        pipeline:
            Base pipeline or ensemble for prediction intervals estimation.
        coverage:
            Interval coverage. In literature this value maybe referred as ``1 - alpha``.
        include_forecast:
            Ensure that the forecast lies within the prediction interval.
        stride:
            Number of points between folds.
        """
        if not (0 <= coverage <= 1):
            raise ValueError("Parameter `coverage` must be non-negative number not greater than 1!")

        if stride <= 0:
            raise ValueError("Parameter `stride` must be positive!")

        self.coverage = coverage
        self.include_forecast = include_forecast
        self.stride = stride

        super().__init__(pipeline=pipeline)

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
        residuals = residuals_matrices(pipeline=self, ts=ts, n_folds=n_folds, stride=self.stride)

        prediction_target = predictions[:, :, "target"]

        levels = 0.5 + np.array([-0.5, 0.5]) * self.coverage
        lower_quantile, upper_quantile = np.quantile(residuals, levels, axis=0)

        # cutoffs to keep prediction inside interval
        if self.include_forecast:
            upper_quantile = np.maximum(upper_quantile, 0)
            lower_quantile = np.minimum(lower_quantile, 0)

        upper_border = prediction_target + upper_quantile
        lower_border = prediction_target + lower_quantile

        upper_border.rename({"target": "target_upper"}, inplace=True, axis=1)
        lower_border.rename({"target": "target_lower"}, inplace=True, axis=1)

        intervals_df = pd.concat([lower_border, upper_border], axis=1)
        predictions.add_prediction_intervals(prediction_intervals_df=intervals_df)
        return predictions
