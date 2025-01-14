from contextlib import suppress
from copy import deepcopy
from typing import List
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import generate_ar_df
from etna.datasets.tsdataset import TSDataset
from etna.datasets.utils import DataFrameFormat
from etna.datasets.utils import apply_alignment
from etna.datasets.utils import infer_alignment
from etna.datasets.utils import make_timestamp_df_from_alignment
from etna.transforms import AddConstTransform
from etna.transforms import DifferencingTransform
from etna.transforms import LagTransform
from etna.transforms import TimeSeriesImputerTransform


@pytest.fixture
def tsdf_with_exog(random_seed) -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-02-01", "2021-07-01", freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-02-01", "2021-07-01", freq="D")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [x**2 + np.random.uniform(-2, 2) for x in list(range(len(df_1)))]
    df_2["segment"] = "Omsk"
    df_2["target"] = [x**0.5 + np.random.uniform(-2, 2) for x in list(range(len(df_2)))]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)

    df = TSDataset.to_dataset(classic_df)

    classic_df_exog = generate_ar_df(start_time="2021-01-01", periods=600, n_segments=2)
    classic_df_exog["segment"] = classic_df_exog["segment"].apply(lambda x: "Moscow" if x == "segment_0" else "Omsk")
    classic_df_exog.rename(columns={"target": "exog"}, inplace=True)
    df_exog = TSDataset.to_dataset(classic_df_exog)

    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    return ts


@pytest.fixture
def tsdf_int_with_exog(tsdf_with_exog) -> TSDataset:
    df = tsdf_with_exog.raw_df
    df_exog = tsdf_with_exog.df_exog
    ref_point = pd.Timestamp("2021-01-01")
    df.index = pd.Index((df.index - ref_point).days, name=df.index.name)
    df_exog.index = pd.Index((df_exog.index - ref_point).days, name=df_exog.index.name)

    ts = TSDataset(df=df, df_exog=df_exog, freq=None)
    return ts


@pytest.fixture
def df_and_regressors() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_1": 3, "regressor_2": 4, "segment": "2"})
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    return df, df_exog, ["regressor_1", "regressor_2"]


@pytest.fixture
def ts_info() -> TSDataset:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df_3 = pd.DataFrame({"timestamp": timestamp, "target": np.NaN, "segment": "3"})
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_1": 3, "regressor_2": 4, "segment": "2"})
    df_3 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 5, "regressor_2": 6, "segment": "3"})
    df_exog = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    # add NaN in the middle
    df.iloc[-5, 0] = np.NaN
    # add NaNs at the end
    df.iloc[-3:, 1] = np.NaN

    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=["regressor_1", "regressor_2"])
    return ts


@pytest.fixture
def ts_info_with_components_and_quantiles() -> TSDataset:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df_3 = pd.DataFrame({"timestamp": timestamp, "target": 13, "segment": "3"})
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df = TSDataset.to_dataset(df)

    ts = TSDataset(df=df, freq="D")

    intervals_df = pd.concat(
        [
            df.rename({"target": "target_0.025"}, axis=1, level="feature") - 1,
            df.rename({"target": "target_0.975"}, axis=1, level="feature") + 1,
        ],
        axis=1,
    )
    ts.add_prediction_intervals(intervals_df)

    components_df = pd.concat(
        [
            df.rename({"target": "target_a"}, axis=1, level="feature") / 2,
            df.rename({"target": "target_b"}, axis=1, level="feature") / 2,
        ],
        axis=1,
    )
    ts.add_target_components(components_df)

    return ts


@pytest.fixture
def df_update_add_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-12")
    df_1 = pd.DataFrame({"timestamp": timestamp, "new_column": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "new_column": 200, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_update_update_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-12")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 200, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_updated_add_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "new_column": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "new_column": 200, "segment": "2"})
    df_2.loc[:4, "target"] = None
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset(df=TSDataset.to_dataset(df), freq="D").df
    return df


@pytest.fixture
def df_updated_update_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 200, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset(df=TSDataset.to_dataset(df), freq="D").df
    return df


@pytest.fixture
def df_exog_updated_add_column() -> pd.DataFrame:
    timestamp = pd.date_range("2020-12-01", "2021-02-12")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "new_column": 100, "segment": "1"})
    df_1.iloc[-1:, df_1.columns.get_loc("regressor_1")] = None
    df_1.iloc[-1:, df_1.columns.get_loc("regressor_2")] = None
    df_1.iloc[:31, df_1.columns.get_loc("new_column")] = None
    df_2 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 3, "regressor_2": 4, "new_column": 200, "segment": "2"})
    df_2.iloc[:5, df_2.columns.get_loc("regressor_1")] = None
    df_2.iloc[:5, df_2.columns.get_loc("regressor_2")] = None
    df_2.iloc[-1:, df_2.columns.get_loc("regressor_1")] = None
    df_2.iloc[-1:, df_2.columns.get_loc("regressor_2")] = None
    df_2.iloc[:31, df_2.columns.get_loc("new_column")] = None
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)
    df_exog = TSDataset(df=df_exog, freq="D").df
    return df_exog


@pytest.fixture
def df_and_regressors_flat() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return flat versions of df and df_exog."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    df_1 = pd.DataFrame(
        {"timestamp": timestamp, "regressor_1": 1, "regressor_2": "3", "regressor_3": 5, "segment": "1"}
    )
    df_2 = pd.DataFrame(
        {"timestamp": timestamp[5:], "regressor_1": 2, "regressor_2": "4", "regressor_3": 6, "segment": "2"}
    )
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog["regressor_2"] = df_exog["regressor_2"].astype("category")
    df_exog["regressor_3"] = df_exog["regressor_3"].astype("category")

    return df, df_exog


@pytest.fixture
def ts_with_categoricals():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2021-01-01", "2021-01-06")
    categorical_values = ["1", "2", "1", "2", "1", "2"]
    df_1 = pd.DataFrame(
        {"timestamp": timestamp, "regressor": categorical_values, "not_regressor": categorical_values, "segment": "1"}
    )
    df_2 = pd.DataFrame(
        {"timestamp": timestamp, "regressor": categorical_values, "not_regressor": categorical_values, "segment": "2"}
    )
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future=["regressor"])
    return ts


@pytest.fixture()
def ts_future(example_reg_tsds):
    future = example_reg_tsds.make_future(10)
    return future


@pytest.fixture
def df_segments_int():
    """DataFrame with integer segments."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 3, "segment": 1})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 4, "segment": 2})
    df = pd.concat([df1, df2], ignore_index=True)
    return df


@pytest.fixture
def target_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 1, "target_component_b": 2, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 3, "target_component_b": 4, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def prediction_intervals_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target_0.1": 1, "target_0.9": 4, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target_0.1": 1, "target_0.9": 10, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def inverse_transformed_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 1 * (3 + 10) / 3,
            "target_component_b": 2 * (3 + 10) / 3,
            "segment": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 3 * (7 + 10) / 7,
            "target_component_b": 4 * (7 + 10) / 7,
            "segment": 2,
        }
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    df.index.freq = "D"
    return df


@pytest.fixture
def inconsistent_target_components_names_df(target_components_df):
    target_components_df = target_components_df.drop(columns=[("2", "target_component_a")])
    return target_components_df


@pytest.fixture
def inconsistent_target_components_names_duplication_df(target_components_df):
    target_components_df = pd.concat(
        (target_components_df, target_components_df.loc[pd.IndexSlice[:], pd.IndexSlice["1", :]]), axis=1
    )
    return target_components_df


@pytest.fixture
def inconsistent_target_components_values_df(target_components_df):
    target_components_df.loc[target_components_df.index[-1], pd.IndexSlice["1", "target_component_a"]] = 100
    target_components_df.loc[target_components_df.index[10], pd.IndexSlice["1", "target_component_a"]] = 100
    return target_components_df


@pytest.fixture
def inconsistent_prediction_intervals_names_df(prediction_intervals_df):
    intervals_df = prediction_intervals_df.drop(columns=[("2", "target_0.1")])
    return intervals_df


@pytest.fixture
def inconsistent_prediction_intervals_names_duplication_df(prediction_intervals_df):
    intervals_df = pd.concat(
        (prediction_intervals_df, prediction_intervals_df.loc[pd.IndexSlice[:], pd.IndexSlice["1", :]]), axis=1
    )
    return intervals_df


@pytest.fixture
def ts_without_target_components():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 3, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 7, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def ts_with_target_components(target_components_df):
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 3, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 7, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df=df)

    ts = TSDataset(df=df, freq="D")
    ts.add_target_components(target_components_df=target_components_df)
    return ts


@pytest.fixture()
def ts_with_prediction_intervals(ts_without_target_components, prediction_intervals_df):
    ts = deepcopy(ts_without_target_components)
    ts.add_prediction_intervals(prediction_intervals_df=prediction_intervals_df)
    return ts


@pytest.fixture()
def ts_after_transform(example_tsds):
    ts = example_tsds
    transform = AddConstTransform(in_column="target", value=0, inplace=False, out_column="add_target")
    ts.fit_transform(transforms=[transform])
    return ts


def test_create_ts_with_datetime_timestamp():
    freq = "D"
    df = generate_ar_df(periods=10, freq=freq, n_segments=3)
    df_wide = TSDataset.to_dataset(df)
    df_wide.index.freq = freq
    ts = TSDataset(df=df_wide, freq=freq)

    pd.testing.assert_index_equal(ts.index, df_wide.index)
    pd.testing.assert_frame_equal(ts.to_pandas(), df_wide)


def test_create_ts_with_int_timestamp():
    df = generate_ar_df(periods=10, freq=None, n_segments=3)
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq=None)

    pd.testing.assert_index_equal(ts.index, df_wide.index)
    pd.testing.assert_frame_equal(ts.to_pandas(), df_wide)


@pytest.mark.filterwarnings(
    "ignore: Timestamp contains numeric values, and given freq is D. Timestamp will be converted to datetime.",
    "ignore: You probably set wrong freq. Discovered freq in you data is N, you set D",
)
def test_create_ts_with_int_timestamp_with_freq():
    df = generate_ar_df(periods=10, freq=None, n_segments=3)
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq="D")

    assert ts.index.dtype == "datetime64[ns]"


def test_create_ts_with_exog_datetime_timestamp():
    freq = "D"
    df = generate_ar_df(periods=10, start_time="2020-01-05", freq=freq, n_segments=3, random_seed=0)
    df_exog = generate_ar_df(periods=20, start_time="2020-01-01", freq=freq, n_segments=3, random_seed=1)
    df_exog.rename(columns={"target": "exog"}, inplace=True)

    df_wide = TSDataset.to_dataset(df)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=freq)

    expected_merged = pd.concat([df_wide, df_exog_wide.loc[df_wide.index]], axis=1).sort_index(axis=1, level=(0, 1))
    expected_merged.index.freq = freq
    pd.testing.assert_index_equal(ts.index, df_wide.index)
    pd.testing.assert_frame_equal(ts.to_pandas(), expected_merged)


def test_create_ts_with_exog_int_timestamp():
    df = generate_ar_df(periods=10, start_time=5, freq=None, n_segments=3, random_seed=0)
    df_exog = generate_ar_df(periods=20, start_time=0, freq=None, n_segments=3, random_seed=1)
    df_exog.rename(columns={"target": "exog"}, inplace=True)

    df_wide = TSDataset.to_dataset(df)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=None)

    expected_merged = pd.concat([df_wide, df_exog_wide.loc[df_wide.index]], axis=1).sort_index(axis=1, level=(0, 1))
    pd.testing.assert_index_equal(ts.index, df_wide.index)
    pd.testing.assert_frame_equal(ts.to_pandas(), expected_merged)


@pytest.mark.filterwarnings(
    "ignore: Timestamp contains numeric values, and given freq is D. Timestamp will be converted to datetime.",
    "ignore: You probably set wrong freq. Discovered freq in you data is N, you set D",
)
def test_create_ts_with_exog_int_timestamp_with_freq():
    df = generate_ar_df(periods=10, start_time=5, freq=None, n_segments=3, random_seed=0)
    df_exog = generate_ar_df(periods=20, start_time=0, freq=None, n_segments=3, random_seed=1)
    df_exog.rename(columns={"target": "exog"}, inplace=True)

    df_wide = TSDataset.to_dataset(df)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq="D")

    assert ts.index.dtype == "datetime64[ns]"


def test_create_ts_missing_datetime_timestamp():
    freq = "D"
    df = generate_ar_df(periods=10, start_time="2020-01-01", freq=freq, n_segments=3, random_seed=0)

    df_wide = TSDataset.to_dataset(df)
    df_wide_missing = df_wide.drop(index=df_wide.index[3:5])
    ts = TSDataset(df=df_wide_missing, freq=freq)

    expected_df = df_wide.copy()
    expected_df.iloc[3:5] = np.NaN
    expected_df.index.freq = freq
    pd.testing.assert_index_equal(ts.index, df_wide.index)
    pd.testing.assert_frame_equal(ts.to_pandas(), expected_df)


def test_create_ts_missing_int_timestamp():
    df = generate_ar_df(periods=10, start_time=5, freq=None, n_segments=3, random_seed=0)

    df_wide = TSDataset.to_dataset(df)
    df_wide_missing = df_wide.drop(index=df_wide.index[3:5])
    ts = TSDataset(df=df_wide_missing, freq=None)

    expected_df = df_wide.copy()
    expected_df.iloc[3:5] = np.NaN
    pd.testing.assert_index_equal(ts.index, df_wide.index)
    pd.testing.assert_frame_equal(ts.to_pandas(), expected_df)


def test_create_ts_with_int_timestamp_fail_datetime():
    df = generate_ar_df(periods=10, freq="D", n_segments=3)
    df_wide = TSDataset.to_dataset(df)
    with pytest.raises(ValueError, match="You set wrong freq"):
        _ = TSDataset(df=df_wide, freq=None)


def test_create_datetime_conversion_during_init():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["categorical_column"] = [0] * 30 + [1] * 30
    classic_df["categorical_column"] = classic_df["categorical_column"].astype("category")
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    df_exog = TSDataset.to_dataset(classic_df[["timestamp", "segment", "categorical_column"]])
    df.index = df.index.astype(str)
    df_exog.index = df.index.astype(str)
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    assert ts.index.dtype == "datetime64[ns]"


def test_create_segment_conversion_during_init(df_segments_int):
    df_wide = TSDataset.to_dataset(df_segments_int)
    df_exog = df_segments_int.rename(columns={"target": "exog"})
    df_exog_wide = TSDataset.to_dataset(df_exog)

    # make conversion back to integers
    columns_frame = df_wide.columns.to_frame()
    columns_frame["segment"] = columns_frame["segment"].astype(int)
    df_wide.columns = pd.MultiIndex.from_frame(columns_frame)

    columns_frame = df_exog_wide.columns.to_frame()
    columns_frame["segment"] = columns_frame["segment"].astype(int)
    df_exog_wide.columns = pd.MultiIndex.from_frame(columns_frame)

    with pytest.warns(UserWarning, match="Segment values doesn't have string type"):
        ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq="D")

    assert np.all(ts.columns.get_level_values("segment") == ["1", "1", "2", "2"])


def test_create_from_long_format_with_exog():
    freq = "D"
    df = generate_ar_df(periods=10, start_time="2020-01-05", freq=freq, n_segments=3, random_seed=0)
    df_exog = generate_ar_df(periods=20, start_time="2020-01-01", freq=freq, n_segments=3, random_seed=1)
    df_exog.rename(columns={"target": "exog"}, inplace=True)
    ts_long = TSDataset(df=df, df_exog=df_exog, freq=freq)

    df_wide = TSDataset.to_dataset(df)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts_wide = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=freq)

    pd.testing.assert_index_equal(ts_long.index, ts_wide.index)
    pd.testing.assert_frame_equal(ts_long.to_pandas(), ts_wide.to_pandas())


@patch("etna.datasets.utils.DataFrameFormat.determine")
def test_create_from_long_format_with_exog_calls_determine(determine_mock):
    determine_mock.side_effect = [DataFrameFormat.long, DataFrameFormat.long]

    freq = "D"
    df = generate_ar_df(periods=10, start_time="2020-01-05", freq=freq, n_segments=3, random_seed=0)
    df_exog = generate_ar_df(periods=20, start_time="2020-01-01", freq=freq, n_segments=3, random_seed=1)
    df_exog.rename(columns={"target": "exog"}, inplace=True)
    _ = TSDataset(df=df, df_exog=df_exog, freq=freq)

    assert determine_mock.call_count == 2


def test_check_endings_error():
    """Check that _check_endings method raises exception if some segments end with nan."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[:-5], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")

    with pytest.raises(ValueError):
        ts._check_endings()


def test_check_endings_pass():
    """Check that _check_endings method passes if there is no nans at the end of all segments."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    ts._check_endings()


def test_check_known_future_wrong_literal():
    """Check that _check_known_future raises exception if wrong literal is given."""
    with pytest.raises(ValueError, match="The only possible literal is 'all'"):
        _ = TSDataset._check_known_future("wrong-literal", None)


def test_check_known_future_error_no_df_exog():
    """Check that _check_known_future raises exception if there are no df_exog, but known_future isn't empty."""
    with pytest.raises(ValueError, match="Some features in known_future are not present in df_exog"):
        _ = TSDataset._check_known_future(["regressor_1"], None)


def test_check_known_future_error_not_matching(df_and_regressors):
    """Check that _check_known_future raises exception if df_exog doesn't contain some features in known_future."""
    _, df_exog, known_future = df_and_regressors
    known_future.append("regressor_new")
    with pytest.raises(ValueError, match="Some features in known_future are not present in df_exog"):
        _ = TSDataset._check_known_future(known_future, df_exog)


def test_check_known_future_pass_all_empty():
    """Check that _check_known_future passes if known_future and df_exog are empty."""
    regressors = TSDataset._check_known_future([], None)
    assert len(regressors) == 0


@pytest.mark.parametrize(
    "known_future, expected_columns",
    [
        ([], []),
        (["regressor_1"], ["regressor_1"]),
        (["regressor_1", "regressor_2"], ["regressor_1", "regressor_2"]),
        (["regressor_1", "regressor_1"], ["regressor_1"]),
        ("all", ["regressor_1", "regressor_2"]),
    ],
)
def test_check_known_future_pass_non_empty(df_and_regressors, known_future, expected_columns):
    _, df_exog, _ = df_and_regressors
    """Check that _check_known_future passes if df_exog is not empty."""
    regressors = TSDataset._check_known_future(known_future, df_exog)
    assert regressors == expected_columns


def test_categorical_after_call_to_pandas():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["categorical_column"] = [0] * 30 + [1] * 30
    classic_df["categorical_column"] = classic_df["categorical_column"].astype("category")
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    exog = TSDataset.to_dataset(classic_df[["timestamp", "segment", "categorical_column"]])
    ts = TSDataset(df, "D", exog)
    flatten_df = ts.to_pandas(flatten=True)
    assert flatten_df["categorical_column"].dtype == "category"


@pytest.mark.parametrize(
    "ts_name, borders, true_borders",
    (
        # datetime timestamp
        (
            "tsdf_with_exog",
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-06-28"),
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-06-28"),
        ),
        (
            "tsdf_with_exog",
            ("2021-02-01", "2021-06-20", "2021-06-23", "2021-07-01"),
            ("2021-02-01", "2021-06-20", "2021-06-23", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            (None, "2021-06-20", "2021-06-23", "2021-06-28"),
            ("2021-02-01", "2021-06-20", "2021-06-23", "2021-06-28"),
        ),
        (
            "tsdf_with_exog",
            ("2021-02-03", "2021-06-20", "2021-06-23", None),
            ("2021-02-03", "2021-06-20", "2021-06-23", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            (None, "2021-06-20", "2021-06-23", None),
            ("2021-02-01", "2021-06-20", "2021-06-23", "2021-07-01"),
        ),
        ("tsdf_with_exog", (None, "2021-06-20", None, None), ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01")),
        ("tsdf_with_exog", (None, None, "2021-06-21", None), ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01")),
        # int timestamp
        (
            "tsdf_int_with_exog",
            (31, 50, 51, 58),
            (31, 50, 51, 58),
        ),
        (
            "tsdf_int_with_exog",
            (35, 50, 51, 58),
            (35, 50, 51, 58),
        ),
        (
            "tsdf_int_with_exog",
            (31, 50, 51, 55),
            (31, 50, 51, 55),
        ),
        (
            "tsdf_int_with_exog",
            (31, 50, 53, 58),
            (31, 50, 53, 58),
        ),
        ("tsdf_int_with_exog", (None, 50, 53, 58), (31, 50, 53, 58)),
        ("tsdf_int_with_exog", (35, 50, 53, None), (35, 50, 53, 181)),
        ("tsdf_int_with_exog", (None, 50, 53, None), (31, 50, 53, 181)),
        ("tsdf_int_with_exog", (None, 50, None, None), (31, 50, 51, 181)),
        ("tsdf_int_with_exog", (None, None, 51, None), (31, 50, 51, 181)),
    ),
)
def test_train_test_split(ts_name, borders, true_borders, request):
    ts = request.getfixturevalue(ts_name)
    train_start, train_end, test_start, test_end = borders
    train_start_true, train_end_true, test_start_true, test_end_true = true_borders
    train, test = ts.train_test_split(
        train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end
    )
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    pd.testing.assert_frame_equal(train.df, ts.df.loc[train_start_true:train_end_true])
    pd.testing.assert_frame_equal(train.df_exog, ts.df_exog)
    pd.testing.assert_frame_equal(test.df, ts.df.loc[test_start_true:test_end_true])
    pd.testing.assert_frame_equal(test.df_exog, ts.df_exog)


@pytest.mark.parametrize(
    "ts_name, test_size, true_borders",
    (
        # datetime timestamp
        ("tsdf_with_exog", 11, ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01")),
        ("tsdf_with_exog", 9, ("2021-02-01", "2021-06-22", "2021-06-23", "2021-07-01")),
        ("tsdf_with_exog", 1, ("2021-02-01", "2021-06-30", "2021-07-01", "2021-07-01")),
        # int timestamp
        ("tsdf_int_with_exog", 11, (31, 170, 171, 181)),
        ("tsdf_int_with_exog", 9, (31, 172, 173, 181)),
        ("tsdf_int_with_exog", 1, (31, 180, 181, 181)),
    ),
)
def test_train_test_split_with_test_size(ts_name, test_size, true_borders, request):
    ts = request.getfixturevalue(ts_name)
    train_start_true, train_end_true, test_start_true, test_end_true = true_borders
    train, test = ts.train_test_split(test_size=test_size)
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    pd.testing.assert_frame_equal(train.df, ts.df.loc[train_start_true:train_end_true])
    pd.testing.assert_frame_equal(train.df_exog, ts.df_exog)
    pd.testing.assert_frame_equal(test.df, ts.df.loc[test_start_true:test_end_true])
    pd.testing.assert_frame_equal(test.df_exog, ts.df_exog)


@pytest.mark.filterwarnings("ignore: test_size, test_start and test_end cannot be")
@pytest.mark.parametrize(
    "ts_name, test_size, borders, true_borders",
    (
        # datetime timestamp
        (
            "tsdf_with_exog",
            10,
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            15,
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            11,
            ("2021-02-02", None, None, "2021-06-28"),
            ("2021-02-02", "2021-06-17", "2021-06-18", "2021-06-28"),
        ),
        (
            "tsdf_with_exog",
            4,
            ("2021-02-03", "2021-06-20", None, "2021-07-01"),
            ("2021-02-03", "2021-06-20", "2021-06-28", "2021-07-01"),
        ),
        (
            "tsdf_with_exog",
            4,
            ("2021-02-03", "2021-06-20", None, None),
            ("2021-02-03", "2021-06-20", "2021-06-21", "2021-06-24"),
        ),
        # int timestamp
        (
            "tsdf_int_with_exog",
            10,
            (31, 171, 172, 181),
            (31, 171, 172, 181),
        ),
        (
            "tsdf_int_with_exog",
            15,
            (31, 169, 172, 181),
            (31, 169, 172, 181),
        ),
        (
            "tsdf_int_with_exog",
            11,
            (33, None, None, 170),
            (33, 159, 160, 170),
        ),
        (
            "tsdf_int_with_exog",
            4,
            (33, 170, None, 181),
            (33, 170, 178, 181),
        ),
        (
            "tsdf_int_with_exog",
            4,
            (33, 170, None, None),
            (33, 170, 171, 174),
        ),
    ),
)
def test_train_test_split_both(ts_name, test_size, borders, true_borders, request):
    ts = request.getfixturevalue(ts_name)
    train_start, train_end, test_start, test_end = borders
    train_start_true, train_end_true, test_start_true, test_end_true = true_borders
    train, test = ts.train_test_split(
        train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, test_size=test_size
    )
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    pd.testing.assert_frame_equal(train.df, ts.df.loc[train_start_true:train_end_true])
    pd.testing.assert_frame_equal(train.df_exog, ts.df_exog)
    pd.testing.assert_frame_equal(test.df, ts.df.loc[test_start_true:test_end_true])
    pd.testing.assert_frame_equal(test.df_exog, ts.df_exog)


@pytest.mark.parametrize(
    "ts_name, borders, match",
    (
        ("tsdf_with_exog", ("2021-01-01", "2021-06-20", "2021-06-21", "2021-07-01"), "Min timestamp in df is"),
        ("tsdf_with_exog", ("2021-02-01", "2021-06-20", "2021-06-21", "2021-08-01"), "Max timestamp in df is"),
        ("tsdf_int_with_exog", (1, 50, 51, 181), "Min timestamp in df is"),
        ("tsdf_int_with_exog", (31, 50, 51, 200), "Max timestamp in df is"),
    ),
)
def test_train_test_split_warning_borders(ts_name, borders, match, request):
    ts = request.getfixturevalue(ts_name)
    train_start, train_end, test_start, test_end = borders
    with pytest.warns(UserWarning, match=match):
        ts.train_test_split(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)


@pytest.mark.parametrize(
    "ts_name, test_size, borders, match",
    (
        (
            "tsdf_with_exog",
            10,
            ("2021-02-01", None, "2021-06-21", "2021-07-01"),
            "test_size, test_start and test_end cannot be applied at the same time. test_size will be ignored",
        ),
        (
            "tsdf_int_with_exog",
            10,
            (31, None, 50, 60),
            "test_size, test_start and test_end cannot be applied at the same time. test_size will be ignored",
        ),
    ),
)
def test_train_test_split_warning_many_parameters(ts_name, test_size, borders, match, request):
    ts = request.getfixturevalue(ts_name)
    train_start, train_end, test_start, test_end = borders
    with pytest.warns(UserWarning, match=match):
        ts.train_test_split(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, test_size=test_size
        )


@pytest.mark.parametrize(
    "ts_name, test_size, borders, match",
    (
        # datetime timestamp
        (
            "tsdf_with_exog",
            None,
            ("2021-02-03", None, None, "2021-07-01"),
            "At least one of train_end, test_start or test_size should be defined",
        ),
        (
            "tsdf_with_exog",
            None,
            (10, "2021-06-20", "2021-06-26", "2021-07-01"),
            "Parameter train_start has incorrect type",
        ),
        (
            "tsdf_with_exog",
            None,
            ("2021-02-03", 10, "2021-06-26", "2021-07-01"),
            "Parameter train_end has incorrect type",
        ),
        (
            "tsdf_with_exog",
            None,
            ("2021-02-03", "2021-06-20", 10, "2021-07-01"),
            "Parameter test_start has incorrect type",
        ),
        (
            "tsdf_with_exog",
            None,
            ("2021-02-03", "2021-06-20", "2021-06-26", 10),
            "Parameter test_end has incorrect type",
        ),
        (
            "tsdf_with_exog",
            17,
            ("2021-02-01", "2021-06-20", None, "2021-07-01"),
            "The beginning of the test goes before the end of the train",
        ),
        (
            "tsdf_with_exog",
            17,
            ("2021-02-01", "2021-06-20", "2021-06-26", None),
            "test_size is 17, but only 6 available with your test_start",
        ),
        # int timestamp
        (
            "tsdf_int_with_exog",
            None,
            (33, None, None, 181),
            "At least one of train_end, test_start or test_size should be defined",
        ),
        (
            "tsdf_int_with_exog",
            None,
            (pd.Timestamp("2020-01-01"), 170, 176, 181),
            "Parameter train_start has incorrect type",
        ),
        (
            "tsdf_int_with_exog",
            None,
            (33, pd.Timestamp("2020-01-01"), 176, 181),
            "Parameter train_end has incorrect type",
        ),
        (
            "tsdf_int_with_exog",
            None,
            (33, 170, pd.Timestamp("2020-01-01"), 181),
            "Parameter test_start has incorrect type",
        ),
        (
            "tsdf_int_with_exog",
            None,
            (33, 170, 176, pd.Timestamp("2020-01-01")),
            "Parameter test_end has incorrect type",
        ),
        (
            "tsdf_int_with_exog",
            17,
            (31, 170, None, 181),
            "The beginning of the test goes before the end of the train",
        ),
        (
            "tsdf_int_with_exog",
            17,
            (31, 50, 176, None),
            "test_size is 17, but only 6 available with your test_start",
        ),
    ),
)
def test_train_test_split_failed(ts_name, test_size, borders, match, request):
    ts = request.getfixturevalue(ts_name)
    train_start, train_end, test_start, test_end = borders
    with pytest.raises(ValueError, match=match):
        ts.train_test_split(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, test_size=test_size
        )


def test_train_test_split_pass_regressors_to_output(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    train, test = ts.train_test_split(test_size=5)
    assert set(train.regressors).issubset(set(train.features))
    assert set(test.regressors).issubset(set(test.features))
    assert train.regressors == ts.regressors
    assert test.regressors == ts.regressors


def test_train_test_split_pass_transform_regressors_to_output(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts.fit_transform(transforms=[LagTransform(in_column="target", lags=[1, 2, 3])])
    train, test = ts.train_test_split(test_size=5)
    assert set(train.regressors).issubset(set(train.features))
    assert set(test.regressors).issubset(set(test.features))
    assert train.regressors == ts.regressors
    assert test.regressors == ts.regressors


def test_train_test_split_pass_target_components_to_output(ts_with_target_components):
    train, test = ts_with_target_components.train_test_split(test_size=5)
    train_target_components = train.get_target_components()
    test_target_components = test.get_target_components()
    assert set(train.target_components_names).issubset(set(train.features))
    assert set(test.target_components_names).issubset(set(test.features))
    assert sorted(train.target_components_names) == sorted(ts_with_target_components.target_components_names)
    assert sorted(test.target_components_names) == sorted(ts_with_target_components.target_components_names)
    assert set(train_target_components.columns.get_level_values("feature")) == set(train.target_components_names)
    assert set(test_target_components.columns.get_level_values("feature")) == set(test.target_components_names)


def test_train_test_split_pass_prediction_intervals_to_output(ts_with_prediction_intervals):
    train, test = ts_with_prediction_intervals.train_test_split(test_size=5)
    train_prediction_intervals = train.get_prediction_intervals()
    test_prediction_intervals = test.get_prediction_intervals()
    assert set(train.prediction_intervals_names).issubset(set(train.features))
    assert set(test.prediction_intervals_names).issubset(set(test.features))
    assert sorted(train.prediction_intervals_names) == sorted(ts_with_prediction_intervals.prediction_intervals_names)
    assert sorted(test.prediction_intervals_names) == sorted(ts_with_prediction_intervals.prediction_intervals_names)
    assert set(train_prediction_intervals.columns.get_level_values("feature")) == set(train.prediction_intervals_names)
    assert set(test_prediction_intervals.columns.get_level_values("feature")) == set(test.prediction_intervals_names)


def test_to_dataset_datetime_conversion():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["timestamp"] = classic_df["timestamp"].astype(str)
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    assert df.index.dtype == "datetime64[ns]"


def test_to_dataset_segment_conversion(df_segments_int):
    """Test that `TSDataset.to_dataset` makes casting of segment to string."""
    df = TSDataset.to_dataset(df_segments_int)
    assert np.all(df.columns.get_level_values("segment") == ["1", "2"])


def test_to_dataset_on_integer_timestamp():
    classic_df = generate_ar_df(periods=30, freq=None, n_segments=2)
    df = TSDataset.to_dataset(classic_df)
    assert pd.api.types.is_integer_dtype(df.index.dtype)


def test_size_with_diff_number_of_features():
    df_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=2, freq="D")
    df_exog_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=1, freq="D")
    df_exog_temp = df_exog_temp.rename({"target": "target_exog"}, axis=1)
    ts_temp = TSDataset(df=TSDataset.to_dataset(df_temp), df_exog=TSDataset.to_dataset(df_exog_temp), freq="D")
    assert ts_temp.size()[0] == len(df_exog_temp)
    assert ts_temp.size()[1] == 2
    assert ts_temp.size()[2] is None


def test_size_target_only():
    df_temp = generate_ar_df(start_time="2023-01-01", periods=40, n_segments=3, freq="D")
    ts_temp = TSDataset(df=TSDataset.to_dataset(df_temp), freq="D")
    assert ts_temp.size()[0] == len(df_temp) / 3
    assert ts_temp.size()[1] == 3
    assert ts_temp.size()[2] == 1


def simple_test_size_():
    df_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=2, freq="D")
    df_exog_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=2, freq="D")
    df_exog_temp = df_exog_temp.rename({"target": "target_exog"}, axis=1)
    df_exog_temp["other_feature"] = 1
    ts_temp = TSDataset(df=TSDataset.to_dataset(df_temp), df_exog=TSDataset.to_dataset(df_exog_temp), freq="D")
    assert ts_temp.size()[0] == len(df_exog_temp) / 2
    assert ts_temp.size()[1] == 2
    assert ts_temp.size()[2] == 3


def test_size_with_diff_number_of_features():
    df_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=2, freq="D")
    df_exog_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=1, freq="D")
    df_exog_temp = df_exog_temp.rename({"target": "target_exog"}, axis=1)
    ts_temp = TSDataset(df=TSDataset.to_dataset(df_temp), df_exog=TSDataset.to_dataset(df_exog_temp), freq="D")
    assert ts_temp.size()[0] == len(df_exog_temp)
    assert ts_temp.size()[1] == 2
    assert ts_temp.size()[2] is None


def test_size_target_only():
    df_temp = generate_ar_df(start_time="2023-01-01", periods=40, n_segments=3, freq="D")
    ts_temp = TSDataset(df=TSDataset.to_dataset(df_temp), freq="D")
    assert ts_temp.size()[0] == len(df_temp) / 3
    assert ts_temp.size()[1] == 3
    assert ts_temp.size()[2] == 1


def simple_test_size_():
    df_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=2, freq="D")
    df_exog_temp = generate_ar_df(start_time="2023-01-01", periods=30, n_segments=2, freq="D")
    df_exog_temp = df_exog_temp.rename({"target": "target_exog"}, axis=1)
    df_exog_temp["other_feature"] = 1
    ts_temp = TSDataset(df=TSDataset.to_dataset(df_temp), df_exog=TSDataset.to_dataset(df_exog_temp), freq="D")
    assert ts_temp.size()[0] == len(df_exog_temp) / 2
    assert ts_temp.size()[1] == 2
    assert ts_temp.size()[2] == 3


@pytest.mark.xfail
def test_make_future_raise_error_on_diff_endings(ts_diff_endings):
    with pytest.raises(ValueError, match="All segments should end at the same timestamp"):
        ts_diff_endings.make_future(10)


def test_make_future_with_imputer(ts_diff_endings, ts_future):
    imputer = TimeSeriesImputerTransform(in_column="target")
    ts_diff_endings.fit_transform([imputer])
    future = ts_diff_endings.make_future(10, transforms=[imputer])
    assert_frame_equal(future.to_pandas(), ts_future.to_pandas())


def test_make_future_datetime_timestamp():
    df = generate_ar_df(periods=20, freq="D", n_segments=2)
    ts = TSDataset(TSDataset.to_dataset(df), freq="D")
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target"}


def test_make_future_int_timestamp():
    freq = None
    df = generate_ar_df(periods=20, freq=freq, n_segments=2)
    ts = TSDataset(TSDataset.to_dataset(df), freq=freq)
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == np.arange(ts.index.max() + 1, ts.index.max() + 10 + 1))
    assert set(ts_future.columns.get_level_values("feature")) == {"target"}


def test_make_future_with_exog_datetime_timestamp(tsdf_with_exog):
    ts = tsdf_with_exog
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target", "exog"}


def test_make_future_with_exog_int_timestamp(tsdf_int_with_exog):
    ts = tsdf_int_with_exog
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == np.arange(ts.index.max() + 1, ts.index.max() + 10 + 1))
    assert set(ts_future.columns.get_level_values("feature")) == {"target", "exog"}


def test_make_future_small_horizon():
    timestamp = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-01"))
    target1 = [np.sin(i) for i in range(len(timestamp))]
    target2 = [np.cos(i) for i in range(len(timestamp))]
    df1 = pd.DataFrame({"timestamp": timestamp, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": target2, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    train = TSDataset(ts[: ts.index[10], :, :], freq="D")
    with pytest.warns(UserWarning, match="TSDataset freq can't be inferred"):
        assert len(train.make_future(1).df) == 1


def test_make_future_with_regressors(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target", "regressor_1", "regressor_2"}


@pytest.mark.parametrize("tail_steps", [11, 0])
def test_make_future_with_regressors_and_context(df_and_regressors, tail_steps):
    df, df_exog, known_future = df_and_regressors
    horizon = 10
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts_future = ts.make_future(horizon, tail_steps=tail_steps)
    assert ts_future.index[tail_steps] == ts.index[-1] + pd.Timedelta("1 day")


def test_make_future_inherits_regressors(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts_future = ts.make_future(10)
    assert ts_future.regressors == ts.regressors


def test_make_future_inherits_hierarchy(product_level_constant_forecast_with_quantiles):
    ts = product_level_constant_forecast_with_quantiles
    future = ts.make_future(future_steps=2)
    assert future.hierarchical_structure is ts.hierarchical_structure


def test_make_future_removes_quantiles(product_level_constant_forecast_with_quantiles):
    ts = product_level_constant_forecast_with_quantiles
    future = ts.make_future(future_steps=2)
    assert len(future.prediction_intervals_names) == 0


def test_make_future_removes_target_components(ts_with_target_components):
    ts = ts_with_target_components
    future = ts.make_future(future_steps=2)
    assert len(future.target_components_names) == 0


def test_make_future_warn_not_enough_regressors(df_and_regressors):
    """Check that warning is thrown if regressors don't have enough values for the future."""
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.warns(UserWarning, match="Some regressors don't have enough values"):
        ts.make_future(ts.df_exog.shape[0] + 100)


@pytest.mark.parametrize("exog_starts_later,exog_ends_earlier", ((True, False), (False, True), (True, True)))
def test_check_regressors_error(exog_starts_later: bool, exog_ends_earlier: bool):
    """Check that error is raised if regressors don't have enough values for the train data."""
    start_time_main = "2021-01-01"
    end_time_main = "2021-02-01"
    start_time_regressors = "2021-01-10" if exog_starts_later else start_time_main
    end_time_regressors = "2021-01-20" if exog_ends_earlier else end_time_main

    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range(start_time_regressors, end_time_regressors)
    df1 = pd.DataFrame({"timestamp": timestamp, "regressor_aaa": 1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_aaa": 2, "segment": "2"})
    df_regressors = pd.concat([df1, df2], ignore_index=True)
    df_regressors = TSDataset.to_dataset(df_regressors)

    with pytest.raises(ValueError):
        TSDataset._check_regressors(df=df, df_regressors=df_regressors)


def test_check_regressors_pass(df_and_regressors):
    """Check that regressors check on creation passes with correct regressors."""
    df, df_exog, _ = df_and_regressors
    _ = TSDataset._check_regressors(df=df, df_regressors=df_exog)


def test_check_regressors_pass_empty(df_and_regressors):
    """Check that regressors check on creation passes with no regressors."""
    df, _, _ = df_and_regressors
    _ = TSDataset._check_regressors(df=df, df_regressors=pd.DataFrame())


def test_getitem_only_date(tsdf_with_exog):
    df_date_only = tsdf_with_exog["2021-02-01"]
    assert df_date_only.name == pd.Timestamp("2021-02-01")
    pd.testing.assert_series_equal(tsdf_with_exog.df.loc["2021-02-01"], df_date_only)


def test_getitem_slice_date(tsdf_with_exog):
    df_slice = tsdf_with_exog["2021-02-01":"2021-02-03"]
    expected_index = pd.DatetimeIndex(pd.date_range("2021-02-01", "2021-02-03"), name="timestamp")
    pd.testing.assert_index_equal(df_slice.index, expected_index)
    pd.testing.assert_frame_equal(tsdf_with_exog.df.loc["2021-02-01":"2021-02-03"], df_slice)


def test_getitem_second_ellipsis(tsdf_with_exog):
    df_slice = tsdf_with_exog["2021-02-01":"2021-02-03", ...]
    expected_index = pd.DatetimeIndex(pd.date_range("2021-02-01", "2021-02-03"), name="timestamp")
    pd.testing.assert_index_equal(df_slice.index, expected_index)
    pd.testing.assert_frame_equal(tsdf_with_exog.df.loc["2021-02-01":"2021-02-03"], df_slice)


def test_getitem_first_ellipsis(tsdf_with_exog):
    df_slice = tsdf_with_exog[..., "target"]
    df_expected = tsdf_with_exog.df.loc[:, [["Moscow", "target"], ["Omsk", "target"]]]
    pd.testing.assert_frame_equal(df_expected, df_slice)


def test_getitem_all_indexes(tsdf_with_exog):
    df_slice = tsdf_with_exog[:, :, :]
    df_expected = tsdf_with_exog.df
    pd.testing.assert_frame_equal(df_expected, df_slice)


def test_finding_regressors_marked(df_and_regressors):
    """Check that ts.regressors property works correctly when regressors set."""
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=["regressor_1", "regressor_2"])
    assert sorted(ts.regressors) == ["regressor_1", "regressor_2"]


def test_finding_regressors_unmarked(df_and_regressors):
    """Check that ts.regressors property works correctly when regressors don't set."""
    df, df_exog, _ = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    assert sorted(ts.regressors) == []


def test_head_default(tsdf_with_exog):
    assert np.all(tsdf_with_exog.head() == tsdf_with_exog.df.head())


def test_tail_default(tsdf_with_exog):
    np.all(tsdf_with_exog.tail() == tsdf_with_exog.df.tail())


def test_right_format_sorting():
    """Need to check if to_dataset method does not mess up with data and column names,
    sorting it with no respect to each other
    """
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100)})
    df["segment"] = "segment_1"
    # need names and values in inverse fashion
    df["reg_2"] = 1
    df["reg_1"] = 2
    tsd = TSDataset(TSDataset.to_dataset(df), freq="D")
    inv_df = tsd.to_pandas(flatten=True)
    pd.testing.assert_series_equal(df["reg_1"], inv_df["reg_1"])
    pd.testing.assert_series_equal(df["reg_2"], inv_df["reg_2"])


def test_to_flatten_simple(example_df):
    """Check that TSDataset.to_flatten works correctly in simple case."""
    flat_df = example_df
    sorted_columns = sorted(flat_df.columns)
    expected_df = flat_df[sorted_columns]
    obtained_df = TSDataset.to_flatten(TSDataset.to_dataset(flat_df))[sorted_columns]
    assert np.all(expected_df.columns == obtained_df.columns)
    assert np.all(expected_df.dtypes == obtained_df.dtypes)
    assert np.all(expected_df.values == obtained_df.values)


def test_to_flatten_simple_int_timestamp():
    flat_df = generate_ar_df(periods=10, freq=None, n_segments=3)
    sorted_columns = sorted(flat_df.columns)
    expected_df = flat_df[sorted_columns]
    obtained_df = TSDataset.to_flatten(TSDataset.to_dataset(flat_df))[sorted_columns]
    assert np.all(expected_df.columns == obtained_df.columns)
    assert np.all(expected_df.dtypes == obtained_df.dtypes)
    assert np.all(expected_df.values == obtained_df.values)


def test_to_flatten_with_exog(df_and_regressors_flat):
    """Check that TSDataset.to_flatten works correctly with exogenous features."""
    df, df_exog = df_and_regressors_flat

    # add boolean dtype
    df_exog["regressor_boolean"] = 1
    df_exog["regressor_boolean"] = df_exog["regressor_boolean"].astype("boolean")
    # add Int64 dtype
    df_exog["regressor_Int64"] = 1
    df_exog.loc[1, "regressor_Int64"] = None
    df_exog["regressor_Int64"] = df_exog["regressor_Int64"].astype("Int64")

    # construct expected result
    flat_df = pd.merge(left=df, right=df_exog, left_on=["timestamp", "segment"], right_on=["timestamp", "segment"])
    sorted_columns = sorted(flat_df.columns)
    expected_df = flat_df[sorted_columns]
    # add values to absent timestamps at one segment
    to_append = pd.DataFrame({"timestamp": df["timestamp"][:5], "segment": ["2"] * 5})
    dtypes = expected_df.dtypes.to_dict()
    expected_df = pd.concat((expected_df, to_append)).sort_values(by=["segment", "timestamp"]).reset_index(drop=True)
    # restore category dtypes: needed for old versions of pandas
    for column, dtype in dtypes.items():
        if dtype == "category":
            expected_df[column] = expected_df[column].astype(dtype)
    # this logic wouldn't work in general case, here we use that all features' names start with 'r'
    sorted_columns = ["timestamp", "segment", "target"] + sorted_columns[:-3]
    # reindex df to assert correct columns order
    expected_df = expected_df[sorted_columns]
    # get to_flatten result
    obtained_df = TSDataset.to_flatten(TSDataset.to_dataset(flat_df))
    pd.testing.assert_frame_equal(obtained_df, expected_df)


@pytest.mark.parametrize(
    "features, expected_columns",
    (
        ("all", ["timestamp", "target", "segment", "regressor_1", "regressor_2"]),
        (["regressor_2"], ["timestamp", "segment", "regressor_2"]),
    ),
)
def test_to_flatten_correct_columns(df_and_regressors, features, expected_columns):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    flattened_df = ts.to_flatten(ts.df, features=features)
    assert sorted(flattened_df.columns) == sorted(expected_columns)


def test_to_flatten_raise_error_incorrect_literal(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.raises(ValueError, match="The only possible literal is 'all'"):
        _ = ts.to_flatten(ts.df, features="incorrect")


def test_to_pandas_simple_int_timestamp():
    df = generate_ar_df(periods=30, freq=None, n_segments=3)
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq=None)
    pandas_df = ts.to_pandas(flatten=False, features="all")
    pd.testing.assert_frame_equal(pandas_df, df_wide)


@pytest.mark.parametrize(
    "features, expected_columns",
    (
        ("all", ["target", "regressor_1", "regressor_2"]),
        (["regressor_2"], ["regressor_2"]),
    ),
)
def test_to_pandas_correct_columns(df_and_regressors, features, expected_columns):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    pandas_df = ts.to_pandas(flatten=False, features=features)
    got_columns = set(pandas_df.columns.get_level_values("feature"))
    assert sorted(got_columns) == sorted(expected_columns)


def test_to_pandas_raise_error_incorrect_literal(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.raises(ValueError, match="The only possible literal is 'all'"):
        _ = ts.to_pandas(flatten=False, features="incorrect")


def test_transform_raise_warning_on_diff_endings(ts_diff_endings):
    with pytest.warns(Warning, match="Segments contains NaNs in the last timestamps."):
        ts_diff_endings.transform([])


def test_fit_transform_raise_warning_on_diff_endings(ts_diff_endings):
    with pytest.warns(Warning, match="Segments contains NaNs in the last timestamps."):
        ts_diff_endings.fit_transform([])


@pytest.mark.parametrize(
    "ts_name, expected_answer",
    [
        ("ts_info", {"num_segments": 3, "num_exogs": 2, "num_regressors": 2, "num_known_future": 2, "freq": "D"}),
        (
            "ts_info_with_components_and_quantiles",
            {"num_segments": 3, "num_exogs": 0, "num_regressors": 0, "num_known_future": 0, "freq": "D"},
        ),
    ],
)
def test_gather_common_data(ts_name, expected_answer, request):
    """Check that TSDataset._gather_common_data correctly finds common data for info/describe methods."""
    ts = request.getfixturevalue(ts_name)
    common_data = ts._gather_common_data()
    assert common_data["num_segments"] == expected_answer["num_segments"]
    assert common_data["num_exogs"] == expected_answer["num_exogs"]
    assert common_data["num_regressors"] == expected_answer["num_regressors"]
    assert common_data["num_known_future"] == expected_answer["num_known_future"]
    assert common_data["freq"] == expected_answer["freq"]


def test_gather_segments_data(ts_info):
    """Check that TSDataset._gather_segments_data correctly finds segment data for info/describe methods."""
    segments_dict = ts_info._gather_segments_data(ts_info.segments)
    segment_df = pd.DataFrame(segments_dict, index=ts_info.segments)

    assert segment_df.loc["1", "start_timestamp"] == pd.Timestamp("2021-01-01")
    assert segment_df.loc["2", "start_timestamp"] == pd.Timestamp("2021-01-06")
    assert segment_df.loc["3", "start_timestamp"] is pd.NaT
    assert segment_df.loc["1", "end_timestamp"] == pd.Timestamp("2021-02-01")
    assert segment_df.loc["2", "end_timestamp"] == pd.Timestamp("2021-01-29")
    assert segment_df.loc["3", "end_timestamp"] is pd.NaT
    assert segment_df.loc["1", "length"] == 32
    assert segment_df.loc["2", "length"] == 24
    assert segment_df.loc["3", "length"] is pd.NA
    assert segment_df.loc["1", "num_missing"] == 1
    assert segment_df.loc["2", "num_missing"] == 0
    assert segment_df.loc["3", "num_missing"] is pd.NA


def test_describe(ts_info):
    """Check that TSDataset.describe works correctly."""
    description = ts_info.describe()

    assert np.all(description.index == ts_info.segments)
    assert description.loc["1", "start_timestamp"] == pd.Timestamp("2021-01-01")
    assert description.loc["2", "start_timestamp"] == pd.Timestamp("2021-01-06")
    assert description.loc["3", "start_timestamp"] is pd.NaT
    assert description.loc["1", "end_timestamp"] == pd.Timestamp("2021-02-01")
    assert description.loc["2", "end_timestamp"] == pd.Timestamp("2021-01-29")
    assert description.loc["3", "end_timestamp"] is pd.NaT
    assert description.loc["1", "length"] == 32
    assert description.loc["2", "length"] == 24
    assert description.loc["3", "length"] is pd.NA
    assert description.loc["1", "num_missing"] == 1
    assert description.loc["2", "num_missing"] == 0
    assert description.loc["3", "num_missing"] is pd.NA
    assert np.all(description["num_segments"] == 3)
    assert np.all(description["num_exogs"] == 2)
    assert np.all(description["num_regressors"] == 2)
    assert np.all(description["num_known_future"] == 2)
    assert np.all(description["freq"] == "D")


@pytest.fixture()
def ts_with_regressors(df_and_regressors):
    df, df_exog, regressors = df_and_regressors
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future="all")
    return ts


def test_to_dataset_not_modify_dataframe():
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_original = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": 1})
    df_copy = df_original.copy(deep=True)
    df_mod = TSDataset.to_dataset(df_original)
    pd.testing.assert_frame_equal(df_original, df_copy)


@pytest.mark.parametrize("start_idx,end_idx", [(1, None), (None, 1), (1, 2), (1, -1)])
def test_tsdataset_idx_slice(tsdf_with_exog, start_idx, end_idx):
    ts_slice = tsdf_with_exog.tsdataset_idx_slice(start_idx=start_idx, end_idx=end_idx)
    assert ts_slice.known_future == tsdf_with_exog.known_future
    assert ts_slice.regressors == tsdf_with_exog.regressors
    pd.testing.assert_frame_equal(ts_slice.df, tsdf_with_exog.df.iloc[start_idx:end_idx])
    pd.testing.assert_frame_equal(ts_slice.df_exog, tsdf_with_exog.df_exog)


def test_tsdataset_idx_slice_pass_target_components_to_output(ts_with_target_components):
    ts_slice = ts_with_target_components.tsdataset_idx_slice(start_idx=1, end_idx=2)
    assert sorted(ts_slice.target_components_names) == sorted(ts_with_target_components.target_components_names)


def test_tsdataset_idx_slice_pass_prediction_intervals_to_output(ts_with_prediction_intervals):
    ts_slice = ts_with_prediction_intervals.tsdataset_idx_slice(start_idx=1, end_idx=2)
    assert sorted(ts_slice.prediction_intervals_names) == sorted(
        ts_with_prediction_intervals.prediction_intervals_names
    )


def test_to_torch_dataset_without_drop(tsdf_with_exog):
    def make_samples(df):
        return [{"target": df.target.values, "segment": df["segment"].values[0]}]

    torch_dataset = tsdf_with_exog.to_torch_dataset(make_samples, dropna=False)
    assert len(torch_dataset) == len(tsdf_with_exog.segments)
    np.testing.assert_array_equal(
        torch_dataset[0]["target"], tsdf_with_exog.df.loc[:, pd.IndexSlice["Moscow", "target"]].values
    )
    np.testing.assert_array_equal(
        torch_dataset[1]["target"], tsdf_with_exog.df.loc[:, pd.IndexSlice["Omsk", "target"]].values
    )


def test_to_torch_dataset_with_drop(tsdf_with_exog):
    def make_samples(df):
        return [{"target": df.target.values, "segment": df["segment"].values[0]}]

    fill_na_idx = tsdf_with_exog.df.index[3]
    tsdf_with_exog.df.loc[:fill_na_idx, pd.IndexSlice["Moscow", "target"]] = np.nan

    torch_dataset = tsdf_with_exog.to_torch_dataset(make_samples, dropna=True)
    assert len(torch_dataset) == len(tsdf_with_exog.segments)
    np.testing.assert_array_equal(
        torch_dataset[0]["target"],
        tsdf_with_exog.df.loc[fill_na_idx + pd.Timedelta("1 day") :, pd.IndexSlice["Moscow", "target"]].values,
    )
    np.testing.assert_array_equal(
        torch_dataset[1]["target"], tsdf_with_exog.df.loc[:, pd.IndexSlice["Omsk", "target"]].values
    )


def test_add_columns_from_pandas_update_df(df_and_regressors, df_update_add_column, df_updated_add_column):
    df, _, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D")
    ts.add_columns_from_pandas(df_update=df_update_add_column, update_exog=False)
    pd.testing.assert_frame_equal(ts.df, df_updated_add_column)


def test_add_columns_from_pandas_update_df_exog(df_and_regressors, df_update_add_column, df_exog_updated_add_column):
    df, df_exog, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D", df_exog=df_exog)
    ts.add_columns_from_pandas(df_update=df_update_add_column, update_exog=True)
    pd.testing.assert_frame_equal(ts.df_exog, df_exog_updated_add_column)


@pytest.mark.parametrize(
    "known_future, regressors, expected_regressors",
    (
        ([], ["regressor_1"], ["regressor_1"]),
        (["regressor_1"], ["regressor_1", "regressor_2"], ["regressor_1", "regressor_2"]),
    ),
)
def test_add_columns_from_pandas_update_regressors(
    df_and_regressors, df_update_add_column, known_future, regressors, expected_regressors
):
    df, df_exog, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future=known_future)
    ts.add_columns_from_pandas(df_update=df_update_add_column, update_exog=True, regressors=regressors)
    assert sorted(ts.regressors) == sorted(expected_regressors)


@pytest.mark.parametrize("update_slice", (slice(4, -4), slice(None, None, 2)))
def test_update_columns_from_pandas_invalid_timestamps(df_and_regressors, update_slice, df_update_update_column):
    df, _, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D")
    with pytest.raises(ValueError, match="Non matching timestamps detected when attempted to update the dataset!"):
        ts.update_columns_from_pandas(df_update=df_update_update_column.iloc[update_slice])


def test_update_columns_from_pandas_invalid_columns_error(df_and_regressors, df_update_update_column):
    df, _, _ = df_and_regressors
    df_update = df_update_update_column.rename({"1": "new"}, axis=1, level=0)
    ts = TSDataset(df=df, freq="D")
    with pytest.raises(ValueError, match="Some columns in the dataframe for update are not presented in the dataset!"):
        ts.update_columns_from_pandas(df_update=df_update)


def test_update_columns_from_pandas_duplicate_columns_error(df_and_regressors, df_update_update_column):
    df, _, _ = df_and_regressors
    df_exog = df.rename(columns={"target": "new"}, level=1)
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    ts.df = pd.concat([ts.df, df_exog], axis=1)

    with pytest.raises(ValueError, match="The dataset features set contains duplicates!"):
        ts.update_columns_from_pandas(df_update=df_update_update_column)


def test_update_columns_from_pandas(df_and_regressors, df_update_update_column, df_updated_update_column):
    df, _, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D")
    ts.update_columns_from_pandas(df_update=df_update_update_column)
    pd.testing.assert_frame_equal(ts.df, df_updated_update_column)


@pytest.mark.filterwarnings("ignore: Features {'out_of_dataset_column'} are not present in")
@pytest.mark.parametrize(
    "features, drop_from_exog, df_expected_columns, df_exog_expected_columns",
    (
        (
            ["regressor_2"],
            False,
            ["timestamp", "segment", "target", "regressor_1"],
            ["timestamp", "segment", "regressor_1", "regressor_2"],
        ),
        (
            ["regressor_2"],
            True,
            ["timestamp", "segment", "target", "regressor_1"],
            ["timestamp", "segment", "regressor_1"],
        ),
        (
            ["regressor_2", "out_of_dataset_column"],
            True,
            ["timestamp", "segment", "target", "regressor_1"],
            ["timestamp", "segment", "regressor_1"],
        ),
    ),
)
def test_drop_features(df_and_regressors, features, drop_from_exog, df_expected_columns, df_exog_expected_columns):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts.drop_features(features=features, drop_from_exog=drop_from_exog)
    df_columns, df_exog_columns = ts.to_flatten(ts.df).columns, ts.to_flatten(ts.df_exog).columns
    assert sorted(df_columns) == sorted(df_expected_columns)
    assert sorted(df_exog_columns) == sorted(df_exog_expected_columns)


def test_drop_features_raise_warning_on_unknown_columns(
    df_and_regressors, features=["regressor_2", "out_of_dataset_column"]
):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.warns(UserWarning, match="Features {'out_of_dataset_column'} are not present in df!"):
        ts.drop_features(features=features, drop_from_exog=False)


@pytest.mark.filterwarnings("ignore: Features {'out_of_dataset_column'} are not present in")
@pytest.mark.parametrize(
    "features, expected_regressors",
    (
        (["regressor_2"], ["regressor_1"]),
        (["out_of_dataset_column"], ["regressor_1", "regressor_2"]),
    ),
)
def test_drop_features_update_regressors(df_and_regressors, features, expected_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts.drop_features(features=features, drop_from_exog=False)
    assert sorted(ts.regressors) == sorted(expected_regressors)


def test_drop_features_throw_error_on_target(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.raises(ValueError, match="Target can't be dropped from the dataset!"):
        ts.drop_features(features=["target"], drop_from_exog=False)


def test_drop_features_throw_error_on_target_components(ts_with_target_components):
    with pytest.raises(
        ValueError,
        match="Target components can't be dropped from the dataset using this method! Use `drop_target_components` method!",
    ):
        ts_with_target_components.drop_features(features=ts_with_target_components.target_components_names)


def test_drop_features_throw_error_on_prediction_intervals(ts_with_prediction_intervals):
    with pytest.raises(
        ValueError,
        match="Prediction intervals can't be dropped from the dataset using this method!",
    ):
        ts_with_prediction_intervals.drop_features(features=ts_with_prediction_intervals.prediction_intervals_names)


def test_get_target_components_on_dataset_without_components(example_tsds):
    target_components_df = example_tsds.get_target_components()
    assert target_components_df is None


def test_get_prediction_intervals_on_dataset_without_components(example_tsds):
    prediction_intervals_df = example_tsds.get_prediction_intervals()
    assert prediction_intervals_df is None


def test_get_target_components(
    ts_with_target_components, expected_components=["target_component_a", "target_component_b"]
):
    expected_target_components_df = ts_with_target_components.to_pandas(features=expected_components)
    target_components_df = ts_with_target_components.get_target_components()
    pd.testing.assert_frame_equal(target_components_df, expected_target_components_df)


def test_get_prediction_intervals(ts_with_prediction_intervals, expected_intervals=["target_0.1", "target_0.9"]):
    expected_prediction_intervals_df = ts_with_prediction_intervals.to_pandas(features=expected_intervals)
    prediction_intervals_df = ts_with_prediction_intervals.get_prediction_intervals()
    pd.testing.assert_frame_equal(prediction_intervals_df, expected_prediction_intervals_df)


def test_add_target_components_throw_error_adding_components_second_time(
    ts_with_target_components, target_components_df
):
    with pytest.raises(ValueError, match="Dataset already contains target components!"):
        ts_with_target_components.add_target_components(target_components_df=target_components_df)


def test_add_prediction_intervals_throw_error_adding_components_second_time(
    ts_with_prediction_intervals, prediction_intervals_df
):
    with pytest.raises(ValueError, match="Dataset already contains prediction intervals!"):
        ts_with_prediction_intervals.add_prediction_intervals(prediction_intervals_df=prediction_intervals_df)


@pytest.mark.parametrize(
    "inconsistent_target_components_names_fixture",
    [("inconsistent_target_components_names_df"), ("inconsistent_target_components_names_duplication_df")],
)
def test_add_target_components_throw_error_inconsistent_components_names(
    ts_without_target_components, inconsistent_target_components_names_fixture, request
):
    inconsistent_target_components_names_df = request.getfixturevalue(inconsistent_target_components_names_fixture)
    with pytest.raises(ValueError, match="Set of target components differs between segments '1' and '2'!"):
        ts_without_target_components.add_target_components(target_components_df=inconsistent_target_components_names_df)


def test_add_target_components_throw_error_inconsistent_components_values(
    ts_without_target_components, inconsistent_target_components_values_df
):
    with pytest.raises(ValueError, match="Components don't sum up to target!"):
        ts_without_target_components.add_target_components(
            target_components_df=inconsistent_target_components_values_df
        )


@pytest.mark.parametrize(
    "inconsistent_prediction_intervals_names_fixture",
    [("inconsistent_prediction_intervals_names_df"), ("inconsistent_prediction_intervals_names_duplication_df")],
)
def test_add_prediction_intervals_throw_error_inconsistent_components_names(
    ts_without_target_components, inconsistent_prediction_intervals_names_fixture, request
):
    inconsistent_prediction_intervals_names_df = request.getfixturevalue(
        inconsistent_prediction_intervals_names_fixture
    )
    with pytest.raises(ValueError, match="Set of prediction intervals differs between segments '1' and '2'!"):
        ts_without_target_components.add_prediction_intervals(
            prediction_intervals_df=inconsistent_prediction_intervals_names_df
        )


def test_add_target_components(ts_without_target_components, ts_with_target_components, target_components_df):
    ts_without_target_components.add_target_components(target_components_df=target_components_df)
    pd.testing.assert_frame_equal(ts_without_target_components.to_pandas(), ts_with_target_components.to_pandas())


def test_add_prediction_intervals(ts_without_target_components, ts_with_prediction_intervals, prediction_intervals_df):
    ts_without_target_components.add_prediction_intervals(prediction_intervals_df=prediction_intervals_df)
    pd.testing.assert_frame_equal(ts_without_target_components.to_pandas(), ts_with_prediction_intervals.to_pandas())


def test_drop_target_components(ts_with_target_components, ts_without_target_components):
    ts_with_target_components.drop_target_components()
    assert ts_with_target_components.target_components_names == ()
    pd.testing.assert_frame_equal(
        ts_with_target_components.to_pandas(),
        ts_without_target_components.to_pandas(),
    )


def test_drop_prediction_intervals(ts_with_prediction_intervals, ts_without_target_components):
    ts_with_prediction_intervals.drop_prediction_intervals()
    assert ts_with_prediction_intervals.prediction_intervals_names == ()
    pd.testing.assert_frame_equal(
        ts_with_prediction_intervals.to_pandas(),
        ts_without_target_components.to_pandas(),
    )


def test_drop_target_components_without_components_in_dataset(ts_without_target_components):
    ts_without_target_components.drop_target_components()
    assert ts_without_target_components.target_components_names == ()


def test_drop_prediction_intervals_without_intervals_in_dataset(ts_without_target_components):
    ts_without_target_components.drop_prediction_intervals()
    assert ts_without_target_components.prediction_intervals_names == ()


def test_inverse_transform_target_components(ts_with_target_components, inverse_transformed_components_df):
    transform = AddConstTransform(in_column="target", value=-10)
    transform.fit(ts=ts_with_target_components)
    ts_with_target_components.inverse_transform([transform])
    assert sorted(ts_with_target_components.target_components_names) == sorted(
        set(inverse_transformed_components_df.columns.get_level_values("feature"))
    )
    pd.testing.assert_frame_equal(ts_with_target_components.get_target_components(), inverse_transformed_components_df)


def test_inverse_transform_with_target_components_fails_keep_target_components(ts_with_target_components):
    transform = DifferencingTransform(in_column="target")
    with suppress(ValueError):
        ts_with_target_components.inverse_transform(transforms=[transform])
    assert len(ts_with_target_components.target_components_names) > 0


@pytest.mark.parametrize(
    "fixture_name, expected_quantiles",
    (("example_tsds", ()), ("product_level_constant_forecast_with_quantiles", ("target_0.25", "target_0.75"))),
)
def test_get_target_quantiles_names(fixture_name, expected_quantiles, request):
    ts = request.getfixturevalue(fixture_name)
    target_quantiles_names = ts.prediction_intervals_names
    assert sorted(target_quantiles_names) == sorted(expected_quantiles)


def test_target_quantiles_names_deprecation_warning(ts_with_prediction_intervals):
    with pytest.warns(
        DeprecationWarning, match="Usage of this property may mislead while accessing prediction intervals."
    ):
        _ = ts_with_prediction_intervals.target_quantiles_names


@pytest.mark.parametrize(
    "ts_name, params, match",
    [
        ("tsdf_with_exog", {"start": 1}, "Parameter start has incorrect type"),
        ("tsdf_with_exog", {"end": 1}, "Parameter end has incorrect type"),
        ("tsdf_int_with_exog", {"start": "2020-01-01"}, "Parameter start has incorrect type"),
        ("tsdf_int_with_exog", {"end": "2020-01-01"}, "Parameter end has incorrect type"),
    ],
)
def test_plot_fail_incorrect_start_end_type(ts_name, params, match, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match=match):
        ts.plot(**params)


@pytest.mark.filterwarnings("ignore: You probably set wrong freq. Discovered freq in you data is N, you set D")
def test_check_timestamp_type_warning():
    match = "Timestamp contains numeric values, and given freq is D. Timestamp will be converted to datetime."

    df = generate_ar_df(periods=10, start_time=5, freq=None, n_segments=3, random_seed=0)
    df_exog = generate_ar_df(periods=20, start_time=0, freq=None, n_segments=3, random_seed=1)
    df_exog.rename(columns={"target": "exog"}, inplace=True)
    df_wide = TSDataset.to_dataset(df)
    df_exog_wide = TSDataset.to_dataset(df_exog)

    with pytest.warns(UserWarning, match=match):
        TSDataset(df=df_wide, freq="D")

    with pytest.warns(UserWarning, match=match):
        TSDataset(df=df_wide, df_exog=df_exog_wide, freq="D")


@pytest.mark.parametrize(
    "df_name, freq, original_timestamp_name, future_steps",
    [
        ("df_aligned_datetime", "D", "external_timestamp", 1),
        ("df_aligned_int", None, "external_timestamp", 1),
        ("df_misaligned_datetime", "D", "external_timestamp", 1),
        ("df_misaligned_int", None, "external_timestamp", 1),
        ("df_misaligned_datetime", "D", "new_timestamp", 1),
        ("df_misaligned_datetime", "D", "external_timestamp", 3),
        ("df_misaligned_datetime", "D", "external_timestamp", 100),
    ],
)
def test_create_from_misaligned_without_exog(df_name, freq, original_timestamp_name, future_steps, request):
    df = request.getfixturevalue(df_name)
    ts = TSDataset.create_from_misaligned(
        df=df, df_exog=None, freq=freq, original_timestamp_name=original_timestamp_name, future_steps=future_steps
    )

    alignment = infer_alignment(df)
    expected_raw_df = TSDataset.to_dataset(apply_alignment(df=df, alignment=alignment))
    pd.testing.assert_frame_equal(ts.raw_df, expected_raw_df)

    timestamp_df = make_timestamp_df_from_alignment(
        alignment=alignment,
        start=expected_raw_df.index[0],
        periods=len(expected_raw_df) + future_steps,
        freq=freq,
        timestamp_name=original_timestamp_name,
    )
    expected_df_exog = TSDataset.to_dataset(timestamp_df)
    pd.testing.assert_frame_equal(ts.df_exog, expected_df_exog)

    assert original_timestamp_name in ts.known_future
    assert ts.freq is None


@pytest.mark.parametrize(
    "df_name, df_exog_name, freq, known_future, original_timestamp_name, future_steps",
    [
        ("df_aligned_datetime", "df_exog_aligned_datetime", "D", ["exog_1"], "external_timestamp", 1),
        ("df_aligned_datetime", "df_exog_misaligned_datetime", "D", ["exog_1"], "external_timestamp", 1),
        ("df_aligned_int", "df_exog_aligned_int", None, ["exog_1"], "external_timestamp", 1),
        ("df_aligned_int", "df_exog_misaligned_int", None, ["exog_1"], "external_timestamp", 1),
        ("df_misaligned_datetime", "df_exog_aligned_datetime", "D", ["exog_1"], "external_timestamp", 1),
        ("df_misaligned_datetime", "df_exog_misaligned_datetime", "D", ["exog_1"], "external_timestamp", 1),
        ("df_misaligned_int", "df_exog_aligned_int", None, ["exog_1"], "external_timestamp", 1),
        ("df_misaligned_int", "df_exog_misaligned_int", None, ["exog_1"], "external_timestamp", 1),
        ("df_misaligned_datetime", "df_exog_misaligned_datetime", "D", ["exog_1"], "new_timestamp", 1),
        ("df_misaligned_datetime", "df_exog_misaligned_datetime", "D", ["exog_1"], "external_timestamp", 3),
        ("df_misaligned_datetime", "df_exog_misaligned_datetime", "D", ["exog_1"], "external_timestamp", 100),
    ],
)
def test_create_from_misaligned_with_exog(
    df_name, df_exog_name, freq, known_future, original_timestamp_name, future_steps, request
):
    df = request.getfixturevalue(df_name)
    df_exog = request.getfixturevalue(df_exog_name)
    ts = TSDataset.create_from_misaligned(
        df=df,
        df_exog=df_exog,
        freq=freq,
        original_timestamp_name=original_timestamp_name,
        future_steps=future_steps,
        known_future=known_future,
    )

    alignment = infer_alignment(df)
    expected_raw_df = TSDataset.to_dataset(apply_alignment(df=df, alignment=alignment))
    pd.testing.assert_frame_equal(ts.raw_df, expected_raw_df)

    expected_df_exog = TSDataset.to_dataset(apply_alignment(df=df_exog, alignment=alignment))
    timestamp_df = make_timestamp_df_from_alignment(
        alignment=alignment,
        start=expected_raw_df.index[0],
        periods=len(expected_raw_df) + future_steps,
        freq=freq,
        timestamp_name=original_timestamp_name,
    )
    expected_df_exog = expected_df_exog.join(TSDataset.to_dataset(timestamp_df), how="outer")
    pd.testing.assert_frame_equal(ts.df_exog, expected_df_exog)

    expected_known_future = sorted(set(known_future).union([original_timestamp_name]))
    assert ts.known_future == expected_known_future

    assert ts.freq is None


@pytest.mark.parametrize(
    "df_name, df_exog_name, freq, original_timestamp_name, future_steps, expected_known_future",
    [
        (
            "df_misaligned_datetime",
            "df_exog_all_misaligned_datetime",
            "D",
            "external_timestamp",
            1,
            ["exog_1", "exog_2", "external_timestamp"],
        ),
    ],
)
def test_create_from_misaligned_with_exog_all(
    df_name, df_exog_name, freq, original_timestamp_name, future_steps, expected_known_future, request
):
    df = request.getfixturevalue(df_name)
    df_exog = request.getfixturevalue(df_exog_name)
    ts = TSDataset.create_from_misaligned(
        df=df,
        df_exog=df_exog,
        freq=freq,
        original_timestamp_name=original_timestamp_name,
        future_steps=future_steps,
        known_future="all",
    )

    alignment = infer_alignment(df)
    expected_raw_df = TSDataset.to_dataset(apply_alignment(df=df, alignment=alignment))
    pd.testing.assert_frame_equal(ts.raw_df, expected_raw_df)

    expected_df_exog = TSDataset.to_dataset(apply_alignment(df=df_exog, alignment=alignment))
    timestamp_df = make_timestamp_df_from_alignment(
        alignment=alignment,
        start=expected_raw_df.index[0],
        periods=len(expected_raw_df) + future_steps,
        freq=freq,
        timestamp_name=original_timestamp_name,
    )
    expected_df_exog = expected_df_exog.join(TSDataset.to_dataset(timestamp_df), how="outer")
    pd.testing.assert_frame_equal(ts.df_exog, expected_df_exog)

    assert ts.known_future == expected_known_future
    assert ts.freq is None


@pytest.mark.parametrize(
    "df_name, freq, original_timestamp_name, future_steps",
    [
        ("df_misaligned_datetime", "D", "external_timestamp", 0),
        ("df_misaligned_datetime", "D", "external_timestamp", -3),
        ("df_misaligned_int", None, "external_timestamp", 0),
        ("df_misaligned_int", None, "external_timestamp", -3),
    ],
)
def test_create_from_misaligned_fail_non_positive_future_steps(
    df_name, freq, original_timestamp_name, future_steps, request
):
    df = request.getfixturevalue(df_name)
    with pytest.raises(ValueError, match="Parameter future_steps should be positive"):
        _ = TSDataset.create_from_misaligned(
            df=df,
            df_exog=None,
            freq=freq,
            original_timestamp_name=original_timestamp_name,
            future_steps=future_steps,
        )


@pytest.mark.parametrize(
    "df_name, df_exog_name, freq, known_future, original_timestamp_name, future_steps",
    [
        ("df_misaligned_datetime", "df_exog_misaligned_datetime", "D", ["exog_1"], "exog_1", 1),
    ],
)
def test_create_from_misaligned_fail_name_intersection(
    df_name, df_exog_name, freq, known_future, original_timestamp_name, future_steps, request
):
    df = request.getfixturevalue(df_name)
    df_exog = request.getfixturevalue(df_exog_name)
    with pytest.raises(
        ValueError, match="Parameter original_timestamp_name shouldn't intersect with columns in df_exog"
    ):
        _ = TSDataset.create_from_misaligned(
            df=df,
            df_exog=df_exog,
            freq=freq,
            original_timestamp_name=original_timestamp_name,
            future_steps=future_steps,
            known_future=known_future,
        )


@pytest.mark.parametrize(
    "ts_name, expected_features",
    [
        ("example_tsds", ["target"]),
        ("tsdf_with_exog", ["target", "exog"]),
        ("ts_after_transform", ["target", "add_target"]),
        ("ts_with_prediction_intervals", ["target", "target_0.1", "target_0.9"]),
        ("ts_with_target_components", ["target", "target_component_a", "target_component_b"]),
    ],
)
def test_features(ts_name, expected_features, request):
    ts = request.getfixturevalue(ts_name)
    features = ts.features
    assert sorted(features) == sorted(expected_features)
