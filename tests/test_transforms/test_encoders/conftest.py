import numpy as np
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def mean_segment_encoder_ts() -> TSDataset:
    df = generate_ar_df(n_segments=2, start_time="2001-01-01", periods=5)
    df["target"] = [0.0, 1.0, np.NaN, 3.0, 4.0] + [np.NaN, 1.0, 2.0, 3.0, 4.0]

    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def expected_mean_segment_encoder_ts() -> TSDataset:
    df = generate_ar_df(n_segments=2, start_time="2001-01-01", periods=5)
    df["target"] = [0.0, 1.0, np.NaN, 3.0, 4.0] + [np.NaN, 1.0, 2.0, 3.0, 4.0]
    df["segment_mean"] = [np.NaN, 0, 0.5, 0.5, 1.33] + [np.NaN, np.NaN, 1, 1.5, 2.0]

    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def expected_make_future_mean_segment_encoder_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-06", periods=2, n_segments=2)
    df["target"] = [np.NaN, np.NaN] + [np.NaN, np.NaN]
    df["segment_mean"] = [2.0, 2.0] + [2.5, 2.5]

    ts = TSDataset(df=df, freq="D")
    return ts
