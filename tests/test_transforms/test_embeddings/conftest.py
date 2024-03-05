import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset


@pytest.fixture
def simple_ts_with_exog() -> TSDataset:
    n_segments = 5
    periods = 10
    timerange = pd.date_range(start="2020-01-01", periods=periods).to_list()
    df = pd.DataFrame({"timestamp": timerange * n_segments})
    segments_list = []
    for i in range(n_segments):
        segments_list += [f"segment_{i}"] * periods
    df["segment"] = segments_list
    df["target"] = (
        [None, None, 3, 4, 5, 6, 7, 8, 9, 10]
        + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        + [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        + [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
    )

    df_exog = pd.DataFrame({"timestamp": timerange * n_segments})
    df_exog["segment"] = segments_list
    df_exog["exog_1"] = df["target"] * 10
    df_exog["exog_2"] = (df["target"] * 3 + 5).astype("category")

    df = TSDataset.to_dataset(df)
    df_exog = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df, freq="D", df_exog=df_exog)
    return ts


@pytest.fixture
def simple_ts_with_exog_numpy(simple_ts_with_exog) -> np.ndarray:
    n_features = 3
    df = simple_ts_with_exog.to_pandas()
    n_timestamps = len(df.index)
    n_segments = df.columns.get_level_values("segment").nunique()
    x = df.values.reshape((n_timestamps, n_segments, n_features)).transpose(1, 0, 2)
    return x
