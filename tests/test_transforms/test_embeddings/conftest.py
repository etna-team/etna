import pandas as pd
import pytest

from etna.datasets import TSDataset


@pytest.fixture
def ts_with_exog_nan_begin() -> TSDataset:
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
def ts_with_exog_nan_middle() -> TSDataset:
    n_segments = 2
    periods = 10
    timerange = pd.date_range(start="2020-01-01", periods=periods).to_list()
    df = pd.DataFrame({"timestamp": timerange * n_segments})
    df["segment"] = ["segment_0"] * periods + ["segment_1"] * periods
    df["target"] = [1, 2, 3, 4, None, None, 7, 8, 9, 10] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    df_exog = pd.DataFrame({"timestamp": timerange * n_segments})
    df_exog["segment"] = ["segment_0"] * periods + ["segment_1"] * periods
    df_exog["exog_1"] = df["target"] * 10

    df = TSDataset.to_dataset(df)
    df_exog = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df, freq="D", df_exog=df_exog)
    return ts


@pytest.fixture
def ts_with_exog_nan_end() -> TSDataset:
    n_segments = 2
    periods = 10
    timerange = pd.date_range(start="2020-01-01", periods=periods).to_list()
    df = pd.DataFrame({"timestamp": timerange * n_segments})
    df["segment"] = ["segment_0"] * periods + ["segment_1"] * periods
    df["target"] = [1, 2, 3, 4, 5, 7, 8, 9, 10, None] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    df = TSDataset.to_dataset(df)

    ts = TSDataset(df=df, freq="D")
    return ts
