from typing import Dict
from typing import Sequence

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import FloatDistribution
from etna.experimental.prediction_intervals import BasePredictionIntervals
from etna.models import CatBoostPerSegmentModel
from etna.models import NaiveModel
from etna.models import SARIMAXModel
from etna.pipeline import BasePipeline
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform


def get_naive_pipeline(horizon):
    return Pipeline(model=NaiveModel(), transforms=[], horizon=horizon)


def get_naive_pipeline_with_transforms(horizon):
    transforms = [AddConstTransform(in_column="target", value=1e6), DateFlagsTransform()]
    return Pipeline(model=NaiveModel(), transforms=transforms, horizon=horizon)


def get_catboost_pipeline(horizon):
    transforms = [LagTransform(in_column="target", lags=list(range(horizon, 2 * horizon)))]
    return Pipeline(model=CatBoostPerSegmentModel(), transforms=transforms, horizon=horizon)


def get_arima_pipeline(horizon):
    return Pipeline(model=SARIMAXModel(order=(1, 0, 1)), horizon=horizon)


def run_base_pipeline_compat_check(ts, intervals_pipeline, expected_columns):
    intervals_pipeline.fit(ts=ts)

    intervals_pipeline_pred = intervals_pipeline.forecast(prediction_interval=True)
    columns = intervals_pipeline_pred._df.columns.get_level_values("feature")

    assert len(expected_columns - set(columns)) == 0
    assert np.sum(intervals_pipeline_pred._df.isna().values) == 0


class DummyPredictionIntervals(BasePredictionIntervals):
    """Dummy class for testing."""

    def __init__(self, pipeline: BasePipeline, width: float = 0.0):
        self.width = width
        super().__init__(pipeline=pipeline)

    def _forecast_prediction_interval(
        self, ts: TSDataset, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Set intervals borders as point forecast."""
        borders = []
        for segment in ts.segments:
            target_df = (predictions[:, segment, "target"]).to_frame()
            target_df.columns.names = predictions._df.columns.names
            borders.append(target_df.rename({"target": f"target_lower"}, axis=1) - self.width / 2)
            borders.append(target_df.rename({"target": f"target_upper"}, axis=1) + self.width / 2)

        # directly store borders in ts._df
        predictions._df = pd.concat([predictions._df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

        return predictions

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        params = super().params_to_tune()
        params["width"] = FloatDistribution(low=-5.0, high=5.0)
        return params
