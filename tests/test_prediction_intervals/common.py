from typing import Sequence

import pandas as pd

from etna.datasets import TSDataset
from etna.pipeline import BasePipeline
from etna.prediction_intervals import BasePredictionIntervals


class DummyPredictionIntervals(BasePredictionIntervals):
    """Dummy class for testing."""

    def __init__(self, pipeline: BasePipeline):
        super().__init__(pipeline=pipeline)

    def _forecast_prediction_interval(
        self, ts: TSDataset, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Set intervals borders as point forecast."""
        borders = []
        for segment in ts.segments:
            target_df = (predictions[:, segment, "target"]).to_frame()
            borders.append(target_df.rename({"target": f"target_lower"}, axis=1))
            borders.append(target_df.rename({"target": f"target_upper"}, axis=1))

        # directly store borders in ts.df
        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

        return predictions
