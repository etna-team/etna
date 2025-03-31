from typing import Optional

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.transforms.timestamp import HolidayTransform
from etna.transforms.timestamp.holiday import define_period
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import convert_ts_to_int_timestamp


@pytest.fixture()
def simple_ts_with_regressors():
    df = generate_const_df(scale=1, n_segments=3, start_time="2020-01-01", periods=100)
    df_exog = generate_const_df(scale=10, n_segments=3, start_time="2020-01-01", periods=150).rename(
        {"target": "regressor_a"}, axis=1
    )
    ts = TSDataset(df=TSDataset.to_dataset(df), freq=pd.offsets.Day(), df_exog=TSDataset.to_dataset(df_exog))
    return ts


@pytest.fixture()
def simple_constant_df_daily():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-01", end="2020-01-15", freq=pd.offsets.Day())})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def simple_constant_df_day_15_min():
    df = pd.DataFrame(
        {"timestamp": pd.date_range(start="2020-11-25 22:30", end="2020-12-11", freq=pd.DateOffset(days=1, minutes=15))}
    )
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

    ts = TSDataset(df=df, df_exog=df_exog, freq=pd.offsets.Day())
    return ts


@pytest.fixture()
def two_segments_simple_ts_daily_int_timestamp(two_segments_simple_ts_daily: TSDataset):
    ts = convert_ts_to_int_timestamp(ts=two_segments_simple_ts_daily)
    return ts


@pytest.fixture
def two_segments_simple_ts_daily_with_regressor(two_segments_simple_ts_daily: TSDataset) -> TSDataset:
    ts = two_segments_simple_ts_daily
    df = ts._raw_df
    df_exog = ts._df_exog
    ts = TSDataset(df=df.iloc[:-3], df_exog=df_exog, freq=ts.freq, known_future=["external_timestamp"])
    return ts


@pytest.fixture()
def two_segments_simple_ts_daily_with_nans(two_segments_simple_ts_daily: TSDataset):
    ts = two_segments_simple_ts_daily
    df = ts._raw_df
    df_exog = ts._df_exog
    df_exog.loc[df_exog.index[:3], pd.IndexSlice[:, "external_timestamp"]] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture()
def two_segments_simple_ts_day_15min(simple_constant_df_day_15_min: pd.DataFrame):
    df_1 = simple_constant_df_day_15_min.reset_index()
    df_2 = simple_constant_df_day_15_min.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.iloc[:3, 0] = np.NaN

    classic_df["external_timestamp"] = classic_df["timestamp"]
    classic_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(classic_df)

    ts = TSDataset(df=df, df_exog=df_exog, freq=pd.DateOffset(days=1, minutes=15))
    return ts


@pytest.fixture()
def simple_constant_df_hour():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-08 22:15", end="2020-01-10", freq=pd.offsets.Hour())})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def simple_week_mon_df():
    df = pd.DataFrame(
        {"timestamp": pd.date_range(start="2020-01-08 22:15", end="2020-05-12", freq=pd.offsets.Week(weekday=0))}
    )
    df["target"] = 7
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_w_mon(simple_week_mon_df: pd.DataFrame):
    df_1 = simple_week_mon_df.reset_index()
    df_2 = simple_week_mon_df.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.iloc[:3, 0] = np.NaN

    classic_df["external_timestamp"] = classic_df["timestamp"]
    classic_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(classic_df)

    ts = TSDataset(df=df, df_exog=df_exog, freq=pd.offsets.Week(weekday=0))
    return ts


@pytest.fixture()
def two_segments_w_mon_int_timestamp(two_segments_w_mon: TSDataset):
    ts = convert_ts_to_int_timestamp(ts=two_segments_w_mon)
    return ts


@pytest.fixture()
def two_segments_w_mon_external_int_timestamp(two_segments_w_mon_int_timestamp: TSDataset):
    ts = two_segments_w_mon_int_timestamp
    df = ts._raw_df
    df_exog = ts._df_exog
    external_int_timestamp = np.arange(len(df_exog))
    df_exog.loc[:, pd.IndexSlice["segment_1", "external_timestamp"]] = external_int_timestamp
    df_exog.loc[:, pd.IndexSlice["segment_2", "external_timestamp"]] = external_int_timestamp
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture()
def two_segments_w_mon_external_irregular_timestamp(two_segments_w_mon: TSDataset):
    ts = two_segments_w_mon
    df = ts._raw_df
    df_exog = ts._df_exog
    df_exog.loc[df_exog.index[3], pd.IndexSlice["segment_1", "external_timestamp"]] += pd.Timedelta("3H")
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture()
def two_segments_w_mon_external_irregular_timestamp_different_freq(two_segments_w_mon: TSDataset):
    ts = two_segments_w_mon
    df = ts._raw_df
    df_exog = ts._df_exog
    df_exog.loc[:, pd.IndexSlice["segment_1", "external_timestamp"]] = pd.date_range(
        start="2020-01-01", periods=len(df_exog), freq=pd.offsets.Week(weekday=6)
    )
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture()
def two_segments_w_mon_with_nans(two_segments_w_mon: TSDataset):
    ts = two_segments_w_mon
    df = ts._raw_df
    df_exog = ts._df_exog
    df_exog.loc[df_exog.index[:3], pd.IndexSlice[:, "external_timestamp"]] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture
def two_segments_w_mon_with_regressor(two_segments_w_mon: TSDataset) -> TSDataset:
    ts = two_segments_w_mon
    df = ts._raw_df
    df_exog = ts._df_exog
    ts = TSDataset(df=df.iloc[:-3], df_exog=df_exog, freq=ts.freq, known_future=["external_timestamp"])
    return ts


@pytest.fixture()
def two_segments_simple_ts_hour(simple_constant_df_hour: pd.DataFrame):
    df_1 = simple_constant_df_hour.reset_index()
    df_2 = simple_constant_df_hour.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq=pd.offsets.Hour())
    return ts


@pytest.fixture()
def simple_constant_df_minute():
    df = pd.DataFrame(
        {"timestamp": pd.date_range(start="2020-11-25 22:30", end="2020-11-26 02:15", freq=pd.offsets.Minute(15))}
    )
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_ts_minute(simple_constant_df_minute):
    df_1 = simple_constant_df_minute.reset_index()
    df_2 = simple_constant_df_minute.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.iloc[:3, 0] = np.NaN

    classic_df["external_timestamp"] = classic_df["timestamp"]
    classic_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(classic_df)

    ts = TSDataset(df=df, df_exog=df_exog, freq=pd.offsets.Minute(15))
    return ts


@pytest.mark.parametrize(
    "freq_offset, timestamp, expected_result",
    (
        (pd.offsets.YearEnd(), pd.Timestamp("2000-12-31"), [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")]),
        (pd.offsets.YearBegin(), pd.Timestamp("2000-01-01"), [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")]),
        (
            pd.offsets.YearEnd(month=10),
            pd.Timestamp("2000-10-31"),
            [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")],
        ),
        (
            pd.offsets.YearBegin(month=10),
            pd.Timestamp("2000-10-01"),
            [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")],
        ),
        (pd.offsets.QuarterEnd(), pd.Timestamp("2000-12-31"), [pd.Timestamp("2000-10-01"), pd.Timestamp("2000-12-31")]),
        (
            pd.offsets.QuarterBegin(startingMonth=1),
            pd.Timestamp("2000-01-01"),
            [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-03-31")],
        ),
        (
            pd.offsets.QuarterEnd(startingMonth=11),
            pd.Timestamp("2000-11-30"),
            [pd.Timestamp("2000-09-01"), pd.Timestamp("2000-11-30")],
        ),
        (
            pd.offsets.QuarterBegin(startingMonth=11),
            pd.Timestamp("2000-11-01"),
            [pd.Timestamp("2000-11-01"), pd.Timestamp("2001-01-31")],
        ),
        (pd.offsets.MonthEnd(), pd.Timestamp("2000-01-31"), [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-31")]),
        (pd.offsets.MonthBegin(), pd.Timestamp("2000-01-01"), [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-31")]),
        (
            pd.offsets.Week(weekday=6),
            pd.Timestamp("2000-12-03"),
            [pd.Timestamp("2000-11-27"), pd.Timestamp("2000-12-03")],
        ),
        (
            pd.offsets.Week(weekday=3),
            pd.Timestamp("2000-11-30"),
            [pd.Timestamp("2000-11-27"), pd.Timestamp("2000-12-03")],
        ),
    ),
)
def test_define_period_end(freq_offset, timestamp, expected_result):
    assert (define_period(freq_offset, timestamp))[0] == expected_result[0]
    assert (define_period(freq_offset, timestamp))[1] == expected_result[1]


def test_fit_days_count_fail_int_index(two_segments_w_mon_int_timestamp):
    ts = two_segments_w_mon_int_timestamp
    transform = HolidayTransform(out_column="holiday", mode="days_count")
    with pytest.raises(ValueError, match="Transform can't work with integer index, parameter in_column should be set"):
        transform.fit(ts=ts)


def test_fit_days_count_fail_external_timestamp_int(two_segments_w_mon_external_int_timestamp):
    ts = two_segments_w_mon_external_int_timestamp
    transform = HolidayTransform(in_column="external_timestamp", out_column="holiday", mode="days_count")
    with pytest.raises(ValueError, match="Transform can work only with datetime external timestamp"):
        transform.fit(ts=ts)


def test_fit_days_count_fail_irregular_timestamp(two_segments_w_mon_external_irregular_timestamp):
    ts = two_segments_w_mon_external_irregular_timestamp
    transform = HolidayTransform(in_column="external_timestamp", out_column="holiday", mode="days_count")
    with pytest.raises(
        ValueError, match="Invalid in_column values! Datetime values should be regular timestamps with some frequency"
    ):
        transform.fit(ts=ts)


def test_fit_days_count_fail_different_freq(two_segments_w_mon_external_irregular_timestamp_different_freq):
    ts = two_segments_w_mon_external_irregular_timestamp_different_freq
    transform = HolidayTransform(in_column="external_timestamp", out_column="holiday", mode="days_count")
    with pytest.raises(
        ValueError, match="Invalid in_column values! Datetime values should have the same frequency for every segment"
    ):
        transform.fit(ts=ts)


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
def test_transform_binary_day(in_column: Optional[str], ts_name, iso_code: str, answer: np.array, request):
    ts = request.getfixturevalue(ts_name)
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday", in_column=in_column)
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)
        assert df[segment]["holiday"].dtype == "category"


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_transform_binary_hour(iso_code: str, answer: np.array, two_segments_simple_ts_hour: TSDataset):
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday")
    df = holidays_finder.fit_transform(two_segments_simple_ts_hour).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)
        assert df[segment]["holiday"].dtype == "category"


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ),
)
def test_transform_binary_minute(iso_code: str, answer: np.array, two_segments_simple_ts_minute):
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday")
    df = holidays_finder.fit_transform(two_segments_simple_ts_minute).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)
        assert df[segment]["holiday"].dtype == "category"


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1])),
        ("US", np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_transform_binary_w_mon(iso_code: str, answer: np.array, two_segments_w_mon):
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="binary", out_column="holiday")
    df = holidays_finder.fit_transform(two_segments_w_mon).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].to_numpy(), answer)
        assert df[segment]["holiday"].dtype == "category"


def test_transform_binary_day_with_nans(two_segments_simple_ts_daily_with_nans):
    ts = two_segments_simple_ts_daily_with_nans
    holidays_finder = HolidayTransform(
        iso_code="RUS", mode="binary", out_column="holiday", in_column="external_timestamp"
    )
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert df[segment]["holiday"].isna().sum() == 3
        assert df[segment]["holiday"].dtype == "category"


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
    [
        ("UK", np.array(["New Year's Day"] + ["NO_HOLIDAY"] * 14)),
        ("US", np.array(["New Year's Day"] + ["NO_HOLIDAY"] * 14)),
    ],
)
def test_transform_category_day(in_column, ts_name, iso_code, answer, request):
    ts = request.getfixturevalue(ts_name)
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="category", out_column="holiday", in_column=in_column)
    df = holidays_finder.fit_transform(ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)
        assert df[segment]["holiday"].dtype == "category"


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        (
            "RUS",
            np.array(
                ["NO_HOLIDAY"] * 6
                + ["Day off (substituted from 02/23/2020)", "NO_HOLIDAY", "Day off (substituted from 03/08/2020)"]
                + ["NO_HOLIDAY"] * 7
                + ["Day off (substituted from 01/04/2020)", "Day off (substituted from 05/09/2020)"]
            ),
        ),
        (
            "US",
            np.array(
                ["NO_HOLIDAY", "Martin Luther King Jr. Day"]
                + ["NO_HOLIDAY"] * 3
                + ["Washington's Birthday"]
                + ["NO_HOLIDAY"] * 12
            ),
        ),
    ),
)
def test_transform_category_w_mon(iso_code: str, answer: np.array, two_segments_w_mon):
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="category", out_column="holiday")
    df = holidays_finder.fit_transform(two_segments_w_mon).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].to_numpy(), answer)
        assert df[segment]["holiday"].dtype == "category"


def test_transform_category_day_with_nans(two_segments_simple_ts_daily_with_nans):
    ts = two_segments_simple_ts_daily_with_nans
    holidays_finder = HolidayTransform(
        iso_code="RUS", mode="category", out_column="holiday", in_column="external_timestamp"
    )
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert df[segment]["holiday"].isna().sum() == 3
        assert df[segment]["holiday"].dtype == "category"


@pytest.mark.xfail()
@pytest.mark.parametrize(
    "in_column, ts_name",
    [
        (None, "two_segments_w_mon"),
        ("external_timestamp", "two_segments_w_mon"),
        ("external_timestamp", "two_segments_w_mon_int_timestamp"),
    ],
)
@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([0, 0, 0, 0, 0, 1 / 7, 0, 1 / 7, 0, 0, 0, 0, 0, 0, 0, 1 / 7, 1 / 7, 0])),
        ("US", np.array([0, 1 / 7, 0, 0, 0, 1 / 7] + 12 * [0])),
    ),
)
def test_transform_days_count_w_mon(in_column, ts_name, iso_code, answer, request):
    ts = request.getfixturevalue(ts_name)
    holidays_finder = HolidayTransform(iso_code=iso_code, mode="days_count", out_column="holiday", in_column=in_column)
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["holiday"].values, answer)


def test_transform_days_count_w_mon_with_nans(two_segments_w_mon_with_nans):
    ts = two_segments_w_mon_with_nans
    holidays_finder = HolidayTransform(
        iso_code="RUS", mode="days_count", out_column="holiday", in_column="external_timestamp"
    )
    ts = holidays_finder.fit_transform(ts)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert df[segment]["holiday"].isna().sum() == 3


@pytest.mark.parametrize(
    "ts_name_fit, ts_name_transform, mode",
    [
        ("two_segments_simple_ts_daily_int_timestamp", "two_segments_simple_ts_daily_int_timestamp", "binary"),
        ("two_segments_simple_ts_daily_int_timestamp", "two_segments_simple_ts_daily_int_timestamp", "category"),
        ("two_segments_w_mon", "two_segments_w_mon_int_timestamp", "days_count"),
    ],
)
def test_transform_fail_int_index(ts_name_fit, ts_name_transform, mode, request):
    ts_fit = request.getfixturevalue(ts_name_fit)
    ts_transform = request.getfixturevalue(ts_name_transform)
    transform = HolidayTransform(out_column="holiday", in_column=None)
    transform.fit(ts_fit)
    with pytest.raises(ValueError, match="Transform can't work with integer index, parameter in_column should be set"):
        _ = transform.transform(ts_transform)


def test_transform_days_count_fail_external_timestamp_int(
    two_segments_w_mon, two_segments_w_mon_external_int_timestamp
):
    ts_fit = two_segments_w_mon
    ts_transform = two_segments_w_mon_external_int_timestamp
    transform = HolidayTransform(in_column="external_timestamp", out_column="holiday", mode="days_count")
    transform.fit(ts_fit)
    with pytest.raises(ValueError, match="Transform can work only with datetime external timestamp"):
        transform.transform(ts=ts_transform)


def test_transform_days_count_fail_irregular_timestamp(
    two_segments_w_mon, two_segments_w_mon_external_irregular_timestamp
):
    ts_fit = two_segments_w_mon
    ts_transform = two_segments_w_mon_external_irregular_timestamp
    transform = HolidayTransform(in_column="external_timestamp", out_column="holiday", mode="days_count")
    transform.fit(ts_fit)
    with pytest.raises(
        ValueError, match="Invalid in_column values! Datetime values should be regular timestamps with some frequency"
    ):
        transform.transform(ts=ts_transform)


def test_transform_days_count_fail_different_freq(
    two_segments_w_mon, two_segments_w_mon_external_irregular_timestamp_different_freq
):
    ts_fit = two_segments_w_mon
    ts_transform = two_segments_w_mon_external_irregular_timestamp_different_freq
    transform = HolidayTransform(in_column="external_timestamp", out_column="holiday", mode="days_count")
    transform.fit(ts_fit)
    with pytest.raises(
        ValueError, match="Invalid in_column values! Datetime values should have the same frequency for every segment"
    ):
        transform.transform(ts=ts_transform)


@pytest.mark.parametrize("ts_name", ("two_segments_simple_ts_daily", "two_segments_simple_ts_minute"))
def test_transform_days_count_mode_fail_wrong_freq(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    holidays_finder = HolidayTransform(out_column="holiday", mode="days_count")
    with pytest.raises(
        ValueError,
        match=f"Days_count mode works only with weekly, monthly, quarterly or yearly data. You have freq={ts.freq}",
    ):
        _ = holidays_finder.fit_transform(ts)


@pytest.mark.parametrize("mode", ["binary", "category", "days_count"])
def test_get_regressors_info_index(mode):
    transform = HolidayTransform(mode=mode, out_column="holiday")

    regressors_info = transform.get_regressors_info()

    expected_regressor_info = ["holiday"]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


@pytest.mark.parametrize("mode", ["binary", "category", "days_count"])
def test_get_regressors_info_in_column_fail_not_fitted(mode):
    transform = HolidayTransform(mode=mode, out_column="holiday", in_column="external_timestamp")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


@pytest.mark.parametrize(
    "ts_name, mode",
    [
        ("two_segments_simple_ts_daily", "binary"),
        ("two_segments_simple_ts_daily", "category"),
        ("two_segments_w_mon", "days_count"),
    ],
)
def test_get_regressors_info_in_column_fitted_exog(ts_name, mode, request):
    ts = request.getfixturevalue(ts_name)
    transform = HolidayTransform(mode=mode, out_column="holiday", in_column="external_timestamp")

    transform.fit(ts)
    regressors_info = transform.get_regressors_info()

    expected_regressor_info = []
    assert sorted(regressors_info) == sorted(expected_regressor_info)


@pytest.mark.parametrize(
    "ts_name, mode",
    [
        ("two_segments_simple_ts_daily_with_regressor", "binary"),
        ("two_segments_simple_ts_daily_with_regressor", "category"),
        ("two_segments_w_mon_with_regressor", "days_count"),
    ],
)
def test_get_regressors_info_in_column_fitted_regressor(ts_name, mode, request):
    ts = request.getfixturevalue(ts_name)
    transform = HolidayTransform(mode=mode, out_column="holiday", in_column="external_timestamp")

    transform.fit(ts)
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
