import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.prediction_intervals.utils import residuals_matrices
from tests.test_prediction_intervals.common import get_naive_pipeline


@pytest.fixture
def constant_ts(size=30) -> TSDataset:
    segment_1 = [7] * size
    segment_2 = [50] * size
    ts_range = list(pd.date_range("2023-01-01", freq=pd.offsets.Day(), periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            "segment": ["segment_1"] * size + ["segment_2"] * size,
        }
    )
    ts = TSDataset(TSDataset.to_dataset(df=df), freq=pd.offsets.Day())
    return ts


@pytest.mark.parametrize("n_folds", (-2, 0))
def test_residuals_matrices_invalid_n_folds(naive_pipeline, example_tsds, n_folds):
    with pytest.raises(ValueError, match="Parameter `n_folds` must be positive!"):
        _ = residuals_matrices(pipeline=naive_pipeline, ts=example_tsds, n_folds=n_folds)


@pytest.mark.parametrize("stride", (-1, 0))
def test_residuals_matrices_invalid_stride(naive_pipeline, example_tsds, stride):
    with pytest.raises(ValueError, match="Parameter `stride` must be positive!"):
        _ = residuals_matrices(pipeline=naive_pipeline, ts=example_tsds, stride=stride)


@pytest.mark.parametrize("horizon,n_folds", ((4, 3), (2, 6), (5, 5)))
def test_residuals_matrices_output_shape(example_tsds, horizon, n_folds):
    pipeline = get_naive_pipeline(horizon=horizon)
    n_segments = len(example_tsds.segments)
    res = residuals_matrices(pipeline=pipeline, ts=example_tsds, n_folds=n_folds)
    assert res.shape == (n_folds, horizon, n_segments)


@pytest.mark.parametrize("stride,n_folds", ((1, 3), (1, 2), (5, 5), (5, 1), (7, 2)))
def test_residuals_matrices_constant_series(naive_pipeline, constant_ts, stride, n_folds):
    res = residuals_matrices(pipeline=naive_pipeline, ts=constant_ts, n_folds=n_folds, stride=stride)
    np.testing.assert_allclose(res, 0)
