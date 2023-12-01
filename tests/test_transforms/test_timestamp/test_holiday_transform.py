from typing import Optional

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.transforms.timestamp import HolidayTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import convert_ts_to_int_timestamp


@pytest.fixture()
def simple_ts_with_regressors():
    df = generate_const_df(scale=1, n_segments=3, start_time="2020-01-01", periods=100)
    df_exog = generate_const_df(scale=10, n_segments=3, start_time="2020-01-01", periods=150).rename(
        {"target": "regressor_a"}, axis=1
    )
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog))
    return ts


@pytest.fixture()
def simple_constant_df_daily():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_ts_daily(simple_constant_df_daily: pd.DataFrame):
    df_1 = simple_constant_df_daily.reset_index()
    df_2 = simple_constant_df_daily.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.iloc[:3, 0] = np.NaN

    classic_df["external_timestamp"] = classic_df["timestamp"]
    classic_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(classic_df)

    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    return ts


@pytest.fixture()
def two_segments_simple_ts_daily_int_timestamp(two_segments_simple_ts_daily: TSDataset):
    ts = convert_ts_to_int_timestamp(ts=two_segments_simple_ts_daily)
    return ts


@pytest.fixture
def two_segments_simple_ts_daily_with_regressor(two_segments_simple_ts_daily: TSDataset) -> TSDataset:
    ts = two_segments_simple_ts_daily
    df = ts.raw_df
    df_exog = ts.df_exog
    ts = TSDataset(df=df.iloc[:-3], df_exog=df_exog, freq=ts.freq, known_future=["external_timestamp"])
    return ts


@pytest.fixture()
def two_segments_simple_ts_daily_with_nans(two_segments_simple_ts_daily: TSDataset):
    ts = two_segments_simple_ts_daily
    df = ts.raw_df
    df_exog = ts.df_exog
    df_exog.loc[df_exog.index[:3], pd.IndexSlice[:, "external_timestamp"]] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture()
def simple_constant_df_hour():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-08 22:15", end="2020-01-10", freq="H")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_ts_hour(simple_constant_df_hour: pd.DataFrame):
    df_1 = simple_constant_df_hour.reset_index()
    df_2 = simple_constant_df_hour.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="H")
    return ts


@pytest.fixture()
def simple_constant_df_minute():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-11-25 22:30", end="2020-11-26 02:15", freq="15T")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_ts_minute(simple_constant_df_minute):
    df_1 = simple_constant_df_minute.reset_index()
    df_2 = simple_constant_df_minute.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="15MIN")
    return ts


@pytest.fixture()
def uk_holiday_names_daily():
    values = ["New Year's Day"] + ["New Year Holiday [Scotland]"] + ["NO_HOLIDAY"] * 13
    return np.array(values)


@pytest.fixture()
def us_holiday_names_daily():
    values = ["New Year's Day"] + ["NO_HOLIDAY"] * 14
    return np.array(values)


def test_holiday_with_regressors(simple_ts_with_regressors: TSDataset):
    holiday = HolidayTransform(out_column="holiday")
    new = holiday.fit_transform(simple_ts_with_regressors)
    len_holiday = len([cols for cols in new.columns if cols[1] == "holiday"])
    assert len_holiday == len(np.unique(new.columns.get_level_values("segment")))


@pytest.mark.parametrize(
    "ts_name", ["two_segments_simple_ts_daily", "two_segments_simple_ts_hour", "two_segments_simple_ts_minute"]
)
def test_transform_interface(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    df = holidays_finder.fit_transform(ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


@pytest.mark.parametrize(
    "in_column, ts_name",
    [
        (None, "two_segments_simple_ts_daily"),
        ("external_timestamp", "two_segments_simple_ts_daily"),
        ("external_timestamp", "two_segments_simple_ts_daily_int_timestamp"),
    ],
)
@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_holidays_binary_day(in_column: Optional[str], ts_name, iso_code: str, answer: np.array, request):
    ts = request.getfixturevalue(ts_name)
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday", in_column=in_column)
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)


def test_holidays_binary_day_with_nans(two_segments_simple_ts_daily_with_nans):
    ts = two_segments_simple_ts_daily_with_nans
    holidays_finder = HolidayTransform(
        iso_code="RUS", mode="binary", out_column="holiday", in_column="external_timestamp"
    )
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert df[segment]["holiday"].isna().sum() == 3


def test_holidays_category_day_with_nans(two_segments_simple_ts_daily_with_nans):
    ts = two_segments_simple_ts_daily_with_nans
    holidays_finder = HolidayTransform(
        iso_code="RUS", mode="category", out_column="holiday", in_column="external_timestamp"
    )
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert df[segment]["holiday"].isna().sum() == 3


@pytest.mark.parametrize(
    "in_column, ts_name",
    [
        (None, "two_segments_simple_ts_daily"),
        ("external_timestamp", "two_segments_simple_ts_daily"),
        ("external_timestamp", "two_segments_simple_ts_daily_int_timestamp"),
    ],
)
@pytest.mark.parametrize(
    "iso_code, answer_name",
    [
        ("UK", "uk_holiday_names_daily"),
        ("US", "us_holiday_names_daily"),
    ],
)
def test_holidays_category_day(in_column, ts_name, iso_code, answer_name, request):
    ts = request.getfixturevalue(ts_name)
    answer = request.getfixturevalue(answer_name)
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="category", out_column="holiday", in_column=in_column)
    df = holidays_finder.fit_transform(ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_holidays_binary_hour(iso_code: str, answer: np.array, two_segments_simple_ts_hour: TSDataset):
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday")
    df = holidays_finder.fit_transform(two_segments_simple_ts_hour).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ),
)
def test_holidays_binary_minute(iso_code: str, answer: np.array, two_segments_simple_ts_minute):
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday")
    df = holidays_finder.fit_transform(two_segments_simple_ts_minute).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)


@pytest.mark.parametrize(
    "index",
    (
        (pd.date_range(start="2020-11-25 22:30", end="2020-12-11", freq="1D 15MIN")),
        (pd.date_range(start="2019-11-25", end="2021-02-25", freq="M")),
    ),
)
def test_holidays_fail_big_frequency(index: pd.DatetimeIndex, two_segments_simple_ts_daily: TSDataset):
    ts = two_segments_simple_ts_daily
    ts.df.index = index
    holidays_finder = HolidayTransform(out_column="holiday")
    with pytest.raises(ValueError, match="Frequency of data should be no more than daily."):
        _ = holidays_finder.fit_transform(ts)


def test_transform_index_fail_int_timestamp(two_segments_simple_ts_daily_int_timestamp):
    transform = HolidayTransform(out_column="holiday", in_column=None)
    transform.fit(two_segments_simple_ts_daily_int_timestamp)
    with pytest.raises(ValueError, match="Transform can't work with integer index, parameter in_column should be set"):
        _ = transform.transform(two_segments_simple_ts_daily_int_timestamp)


@pytest.mark.parametrize("mode", ["binary", "category"])
def test_get_regressors_info_index(mode, two_segments_simple_ts_daily):
    transform = HolidayTransform(mode=mode, out_column="holiday")

    regressors_info = transform.get_regressors_info()

    expected_regressor_info = ["holiday"]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


@pytest.mark.parametrize("mode", ["binary", "category"])
def test_get_regressors_info_in_column_fail_not_fitted(mode, two_segments_simple_ts_daily):
    transform = HolidayTransform(mode=mode, out_column="holiday", in_column="external_timestamp")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


@pytest.mark.parametrize("mode", ["binary", "category"])
def test_get_regressors_info_in_column_fitted_exog(mode, two_segments_simple_ts_daily):
    transform = HolidayTransform(mode=mode, out_column="holiday", in_column="external_timestamp")

    transform.fit(two_segments_simple_ts_daily)
    regressors_info = transform.get_regressors_info()

    expected_regressor_info = []
    assert sorted(regressors_info) == sorted(expected_regressor_info)


@pytest.mark.parametrize("mode", ["binary", "category"])
def test_get_regressors_info_in_column_fitted_regressor(mode, two_segments_simple_ts_daily_with_regressor):
    transform = HolidayTransform(mode=mode, out_column="holiday", in_column="external_timestamp")

    transform.fit(two_segments_simple_ts_daily_with_regressor)
    regressors_info = transform.get_regressors_info()

    expected_regressor_info = ["holiday"]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


@pytest.mark.parametrize("expected_regressors", ([["holiday"]]))
def test_holidays_out_column_added_to_regressors(example_tsds, expected_regressors):
    holidays_finder = HolidayTransform(out_column="holiday")
    example_tsds = holidays_finder.fit_transform(example_tsds)
    assert sorted(example_tsds.regressors) == sorted(expected_regressors)


def test_save_load(example_tsds):
    transform = HolidayTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)


def test_params_to_tune():
    transform = HolidayTransform()
    assert len(transform.params_to_tune()) == 0
