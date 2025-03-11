import numpy as np
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture()
def ts_with_exogs() -> TSDataset:
    periods = 100
    periods_exog = periods + 10
    df = generate_ar_df(start_time="2020-01-01", periods=periods, freq="D", n_segments=2)
    df_exog = generate_ar_df(start_time="2020-01-01", periods=periods_exog, freq="D", n_segments=2, random_seed=2)
    df_exog.rename(columns={"target": "exog"}, inplace=True)
    df_exog["holiday"] = np.random.choice([0, 1], size=periods_exog * 2)

    ts = TSDataset(df, freq="D", df_exog=df_exog, known_future="all")
    return ts


@pytest.fixture()
def ts_with_exogs_train_test(ts_with_exogs):
    return ts_with_exogs.train_test_split(test_size=20)


@pytest.fixture()
def forward_stride_datasets(ts_with_exogs):
    train_df = ts_with_exogs._df.iloc[:-10]
    test_df = ts_with_exogs._df.iloc[-20:]

    train_ts = TSDataset(df=train_df, freq=ts_with_exogs.freq)
    test_ts = TSDataset(df=test_df, freq=ts_with_exogs.freq)

    return train_ts, test_ts
