import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data


@pytest.fixture
def regular_ts(random_seed) -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_1["segment"] = "segment_1"
    df_1["target"] = np.random.uniform(10, 20, size=periods)

    df_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_2["segment"] = "segment_2"
    df_2["target"] = np.random.uniform(-15, 5, size=periods)

    df_3 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_3["segment"] = "segment_3"
    df_3["target"] = np.random.uniform(-5, 5, size=periods)

    df = pd.concat([df_1, df_2, df_3]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq=pd.offsets.Day())

    return tsds


@pytest.fixture
def regular_ts_one_month(random_seed) -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods, freq=pd.offsets.MonthEnd())})
    df_1["segment"] = "segment_1"
    df_1["target"] = np.random.uniform(10, 20, size=periods)

    df_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods, freq=pd.offsets.MonthEnd())})
    df_2["segment"] = "segment_2"
    df_2["target"] = np.random.uniform(-15, 5, size=periods)

    df_3 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods, freq=pd.offsets.MonthEnd())})
    df_3["segment"] = "segment_3"
    df_3["target"] = np.random.uniform(-5, 5, size=periods)

    df = pd.concat([df_1, df_2, df_3]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq=pd.offsets.MonthEnd())

    return tsds


@pytest.fixture
def ts_with_exog(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas(flatten=True)
    periods = 200
    timestamp = pd.date_range("2020-01-01", periods=periods)
    df_exog_common = pd.DataFrame(
        {
            "timestamp": timestamp,
            "positive": 1.0,
            "weekday": timestamp.weekday,
            "monthday": timestamp.day,
            "month": timestamp.month,
            "year": timestamp.year,
        }
    )
    df_exog_wide = duplicate_data(df=df_exog_common, segments=regular_ts.segments)

    rng = np.random.default_rng(1)
    df_exog_wide.loc[:, pd.IndexSlice["segment_1", "positive"]] = rng.uniform(5, 10, size=periods)
    df_exog_wide.loc[:, pd.IndexSlice["segment_2", "positive"]] = rng.uniform(5, 10, size=periods)
    df_exog_wide.loc[:, pd.IndexSlice["segment_3", "positive"]] = rng.uniform(5, 10, size=periods)

    ts = TSDataset(df=TSDataset.to_dataset(df).iloc[5:], df_exog=df_exog_wide, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_with_exog_to_shift(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas(flatten=True)
    periods = 120
    timestamp = pd.date_range("2020-01-01", periods=periods)
    feature = timestamp.weekday.astype(float)
    df_exog_common = pd.DataFrame(
        {
            "timestamp": timestamp,
            "feature_1": feature[:100].tolist() + [None] * 20,
            "feature_2": feature[:105].tolist() + [None] * 15,
            "feature_3": feature,
        }
    )
    df_exog_wide = duplicate_data(df=df_exog_common, segments=regular_ts.segments)
    ts = TSDataset(df=TSDataset.to_dataset(df).iloc[5:], df_exog=df_exog_wide, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_with_external_timestamp(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas(flatten=True)
    df_exog = df.copy()
    df_exog["external_timestamp"] = df["timestamp"]
    df_exog.drop(columns=["target"], inplace=True)
    ts = TSDataset(
        df=TSDataset.to_dataset(df).iloc[1:-10],
        df_exog=TSDataset.to_dataset(df_exog),
        freq=pd.offsets.Day(),
        known_future="all",
    )
    return ts


@pytest.fixture
def ts_with_external_timestamp_one_month(regular_ts_one_month) -> TSDataset:
    df = regular_ts_one_month.to_pandas(flatten=True)
    df_exog = df.copy()
    df_exog["external_timestamp"] = df["timestamp"]
    df_exog.drop(columns=["target"], inplace=True)
    ts = TSDataset(
        df=TSDataset.to_dataset(df).iloc[1:-10],
        df_exog=TSDataset.to_dataset(df_exog),
        freq=pd.offsets.MonthEnd(),
        known_future="all",
    )
    return ts


@pytest.fixture
def ts_with_external_int_timestamp(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas(flatten=True)
    df_exog = df.copy()
    df_exog["external_timestamp"] = np.arange(10, 110).tolist() * 3
    df_exog.drop(columns=["target"], inplace=True)
    ts = TSDataset(
        df=TSDataset.to_dataset(df).iloc[1:-10],
        df_exog=TSDataset.to_dataset(df_exog),
        freq=pd.offsets.Day(),
        known_future="all",
    )
    return ts


@pytest.fixture
def positive_ts() -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq=pd.offsets.Day())})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq=pd.offsets.Day())})
    df_3 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq=pd.offsets.Day())})
    generator = np.random.RandomState(seed=1)

    df_1["segment"] = "segment_1"
    df_1["target"] = np.abs(generator.normal(loc=10, scale=1, size=len(df_1))) + 1

    df_2["segment"] = "segment_2"
    df_2["target"] = np.abs(generator.normal(loc=20, scale=1, size=len(df_2))) + 1

    df_3["segment"] = "segment_3"
    df_3["target"] = np.abs(generator.normal(loc=30, scale=1, size=len(df_2))) + 1

    classic_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    wide_df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=wide_df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_to_fill(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas()
    df.iloc[5, 0] = np.NaN
    df.iloc[10, 1] = np.NaN
    df.iloc[20, 2] = np.NaN
    df.iloc[-5, 0] = np.NaN
    df.iloc[-10, 1] = np.NaN
    df.iloc[-20, 2] = np.NaN
    ts = TSDataset(df=df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_to_resample() -> TSDataset:
    df_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq=pd.offsets.Hour(), periods=120),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq=pd.offsets.Hour(), periods=120),
            "segment": "segment_2",
            "target": ([1] + 23 * [0]) * 5,
        }
    )
    df_3 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq=pd.offsets.Hour(), periods=120),
            "segment": "segment_3",
            "target": ([4] + 23 * [0]) * 5,
        }
    )
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)

    df_exog_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq=pd.offsets.Day(), periods=8),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq=pd.offsets.Day(), periods=8),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog_3 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq=pd.offsets.Day(), periods=8),
            "segment": "segment_3",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog_1, df_exog_2, df_exog_3], ignore_index=True)
    ts = TSDataset(
        df=TSDataset.to_dataset(df), freq=pd.offsets.Hour(), df_exog=TSDataset.to_dataset(df_exog), known_future="all"
    )
    return ts


@pytest.fixture
def ts_to_resample_int_timestamp() -> TSDataset:
    df_1 = pd.DataFrame(
        {
            "timestamp": np.arange(24, 144),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": np.arange(24, 144),
            "segment": "segment_2",
            "target": ([1] + 23 * [0]) * 5,
        }
    )
    df_3 = pd.DataFrame(
        {
            "timestamp": np.arange(24, 144),
            "segment": "segment_3",
            "target": ([4] + 23 * [0]) * 5,
        }
    )
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)

    df_exog_1 = pd.DataFrame(
        {
            "timestamp": np.arange(24, 216, 24),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog_2 = pd.DataFrame(
        {
            "timestamp": np.arange(24, 216, 24),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog_3 = pd.DataFrame(
        {
            "timestamp": np.arange(24, 216, 24),
            "segment": "segment_3",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog_1, df_exog_2, df_exog_3], ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq=None, df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    return ts


@pytest.fixture
def ts_with_outliers(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas()
    df.iloc[5, 0] *= 100
    df.iloc[10, 1] *= 100
    df.iloc[20, 2] *= 100
    df.iloc[-5, 0] *= 100
    df.iloc[-10, 1] *= 100
    df.iloc[-20, 2] *= 100
    ts = TSDataset(df=df, freq=pd.offsets.Day())
    return ts
