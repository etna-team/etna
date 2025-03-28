from typing import Sequence

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.pipeline import BasePipeline
from etna.prediction_intervals import BasePredictionIntervals
from etna.prediction_intervals.utils import residuals_matrices


class ConformalPredictionIntervals(BasePredictionIntervals):
    """Estimate conformal prediction intervals using absolute values of historical residuals.

    1. Compute matrix of absolute residuals  :math:`r_{it} = |\hat y_{it} - y_{it}|` using k-fold backtest, where :math:`i` is fold index.

    2. Estimate corresponding quantiles levels using the provided coverage (e.g. apply Bonferroni correction).

    3. Estimate quantiles for each timestamp using computed absolute residuals and levels.

    `Relevant paper <https://proceedings.neurips.cc/paper/2021/file/312f1ba2a72318edaaa995a67835fad5-Paper.pdf>`_.
    `Reference implementation <https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.conformal.ConformalIntervals.html>`_.
    """

    def __init__(
        self, pipeline: BasePipeline, coverage: float = 0.95, bonferroni_correction: bool = False, stride: int = 1
    ):
        """Initialize instance of ``ConformalPredictionIntervals`` with given parameters.

        Parameters
        ----------
        pipeline:
            Base pipeline or ensemble for prediction intervals estimation.
        coverage:
             Interval coverage. In literature this value maybe referred as ``1 - alpha``.
        bonferroni_correction:
             Whether to use Bonferroni correction when estimating quantiles.
        stride:
            Number of points between folds.
        """
        if not (0 <= coverage <= 1):
            raise ValueError("Parameter `coverage` must be non-negative number not greater than 1!")

        if stride <= 0:
            raise ValueError("Parameter `stride` must be positive!")

        self.coverage = coverage
        self.bonferroni_correction = bonferroni_correction
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
        abs_residuals = np.abs(residuals)

        level = self.coverage
        if self.bonferroni_correction:
            level = 1 - (1 - self.coverage) / self.horizon

        critical_scores = np.quantile(abs_residuals, q=level, axis=0)

        upper_border = predictions[:, :, "target"] + critical_scores
        upper_border.rename({"target": "target_upper"}, inplace=True, axis=1)

        lower_border = predictions[:, :, "target"] - critical_scores
        lower_border.rename({"target": "target_lower"}, inplace=True, axis=1)

        intervals_df = pd.concat([lower_border, upper_border], axis=1)
        predictions.add_prediction_intervals(prediction_intervals_df=intervals_df)
        return predictions
