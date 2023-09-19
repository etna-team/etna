from typing import Sequence

import pandas as pd

from etna.datasets import TSDataset
from etna.experimental.prediction_intervals import BasePredictionIntervals
from etna.experimental.prediction_intervals.mixins import SavePredictionIntervalsMixin
from etna.models import NaiveModel
from etna.pipeline import BasePipeline
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform


def get_naive_pipeline(horizon):
    return Pipeline(model=NaiveModel(), transforms=[], horizon=horizon)


def get_naive_pipeline_with_transforms(horizon):
    transforms = [AddConstTransform(in_column="target", value=1e6), DateFlagsTransform()]
    return Pipeline(model=NaiveModel(), transforms=transforms, horizon=horizon)


class DummyPredictionIntervals(SavePredictionIntervalsMixin, BasePredictionIntervals):
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
            borders.append(target_df.rename({"target": f"target_lower"}, axis=1) - self.width / 2)
            borders.append(target_df.rename({"target": f"target_upper"}, axis=1) + self.width / 2)

        # directly store borders in ts.df
        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

        return predictions
