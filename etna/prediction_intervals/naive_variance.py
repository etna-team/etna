from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats as scs

from etna.datasets import TSDataset
from etna.pipeline import BasePipeline
from etna.prediction_intervals import BasePredictionIntervals
from etna.prediction_intervals.utils import residuals_matrices


class NaiveVariancePredictionIntervals(BasePredictionIntervals):
    """Estimate prediction variance based on historical residuals.

    ``NaiveVariancePredictionIntervals`` provides the possibility to estimate prediction quantiles using the following algorithm:

    1. Compute the residuals matrix :math:`r_{it} = \hat y_{it} - y_{it}` using k-fold backtest, where :math:`i` is fold index.

    2. Estimate variance for each step in the prediction horizon :math:`v_t = \\frac{1}{k} \sum_{i = 1}^k r_{it}^2`.

    3. Use :math:`z` scores and estimated variance to compute corresponding quantiles.

    `Reference implementation <https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.naive.NaiveVariance.html>`_.
    """

    def __init__(self, pipeline: BasePipeline, stride: int = 1):
        """Initialize instance of ``NaiveVariancePredictionIntervals`` with given parameters.

        Parameters
        ----------
        pipeline:
            Base pipeline or ensemble for prediction intervals estimation.
        stride:
            Number of points between folds.
        """
        if stride <= 0:
            raise ValueError("Parameter `stride` must be positive!")

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

        variance = self._estimate_variance(residual_matrices=residuals)

        borders = []
        for q in quantiles:
            z_score = scs.norm.ppf(q=q)
            interval_border = predictions[:, :, "target"] + np.sqrt(variance) * z_score
            interval_border.rename({"target": f"target_{q:.4g}"}, inplace=True, axis=1)
            borders.append(interval_border)

        quantiles_df = pd.concat(borders, axis=1)
        predictions.add_prediction_intervals(prediction_intervals_df=quantiles_df)
        return predictions

    def _estimate_variance(self, residual_matrices: np.ndarray) -> np.ndarray:
        """Estimate variance from residuals matrices.

        Parameters
        ----------
        residual_matrices:
            Multidimensional array with shape ``(n_folds, horizon, n_segments)``.

        Returns
        -------
        :
            Estimated variance. Array with shape ``(horizon, n_segments)``.

        """
        variance = np.mean(np.power(residual_matrices, 2), axis=0)
        return variance
