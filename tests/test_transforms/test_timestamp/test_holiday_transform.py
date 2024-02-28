import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.transforms.timestamp import HolidayTransform
from etna.transforms.timestamp.holiday import define_period
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


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
def simple_constant_df_day_15_min():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-11-25 22:30", end="2020-12-11", freq="1D 15MIN")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_ts_daily(simple_constant_df_daily: pd.DataFrame):
    df_1 = simple_constant_df_daily.reset_index()
    df_2 = simple_constant_df_daily.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture()
def two_segments_simple_ts_day_15min(simple_constant_df_day_15_min: pd.DataFrame):
    df_1 = simple_constant_df_day_15_min.reset_index()
    df_2 = simple_constant_df_day_15_min.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="1D 15MIN")
    return ts


@pytest.fixture()
def simple_constant_df_hour():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-08 22:15", end="2020-01-10", freq="H")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def simple_week_mon_df():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-08 22:15", end="2020-05-12", freq="W-MON")})
    df["target"] = 7
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def simple_q_jan_df_():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-08 22:15", end="2021-01-10", freq="Q")})
    df["target"] = 90
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_w_mon(simple_week_mon_df: pd.DataFrame):
    df_1 = simple_week_mon_df.reset_index()
    df_2 = simple_week_mon_df.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="W-MON")
    return ts


@pytest.fixture()
def two_segments_q_feb(simple_q_jan_df_: pd.DataFrame):
    df_1 = simple_q_jan_df_.reset_index()
    df_2 = simple_q_jan_df_.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="Q")
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
    ts = TSDataset(df, freq="H")
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
    ts = TSDataset(df, freq="H")
    return ts


@pytest.fixture()
def simple_constant_df_min():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-11-25 22:30", end="2020-11-26 02:15", freq="15MIN")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_ts_min(simple_constant_df_min: pd.DataFrame):
    df_1 = simple_constant_df_min.reset_index()
    df_2 = simple_constant_df_min.reset_index()
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


def test_interface_two_segments_daily(two_segments_simple_ts_daily: TSDataset):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_daily)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


def test_interface_two_segments_hour(two_segments_simple_ts_hour: TSDataset):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_hour)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


def test_interface_two_segments_min(two_segments_simple_ts_min: TSDataset):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_min)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_holidays_day(iso_code: str, answer: np.array, two_segments_simple_ts_daily: TSDataset):
    holidays_finder = HolidayTransform(iso_code=iso_code, out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_daily)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, answer)


def test_uk_holidays_day_category(uk_holiday_names_daily: np.array, two_segments_simple_ts_daily: TSDataset):
    holidays_finder = HolidayTransform(iso_code="UK", mode="category", out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_daily)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, uk_holiday_names_daily)


def test_us_holidays_day_category(us_holiday_names_daily: np.array, two_segments_simple_ts_daily: TSDataset):
    holidays_finder = HolidayTransform(iso_code="US", mode="category", out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_daily)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, us_holiday_names_daily)


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_holidays_hour(iso_code: str, answer: np.array, two_segments_simple_ts_hour: TSDataset):
    holidays_finder = HolidayTransform(iso_code=iso_code, out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_hour)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, answer)


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ),
)
def test_holidays_min(iso_code: str, answer: np.array, two_segments_simple_ts_min: TSDataset):
    holidays_finder = HolidayTransform(iso_code=iso_code, out_column="regressor_holidays")
    ts = holidays_finder.fit_transform(two_segments_simple_ts_min)
    df = ts.to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, answer)


@pytest.mark.parametrize(
    "index",
    (
        (pd.date_range(start="2020-11-25 22:30", end="2020-12-11", freq="1D 15MIN")),
        (pd.date_range(start="2019-11-25", end="2021-02-25", freq="M")),
    ),
)
def test_holidays_failed(index: pd.DatetimeIndex, two_segments_simple_ts_day_15min: TSDataset):
    ts = two_segments_simple_ts_day_15min
    ts.df.index = index
    holidays_finder = HolidayTransform(out_column="holiday")
    with pytest.raises(
        ValueError, match="For binary and category modes frequency of data should be no more than daily."
    ):
        ts = holidays_finder.fit_transform(ts)


def test_holidays_days_count_mode_failed(two_segments_simple_ts_daily: TSDataset):
    ts = two_segments_simple_ts_daily
    holidays_finder = HolidayTransform(out_column="holiday", mode="days_count")
    with pytest.raises(
        ValueError, match="Days_count mode works only with weekly, monthly, quarterly or yearly data. You have freq=D"
    ):
        ts = holidays_finder.fit_transform(ts)


@pytest.mark.parametrize("expected_regressors", ([["regressor_holidays"]]))
def test_holidays_out_column_added_to_regressors(example_tsds, expected_regressors):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    example_tsds = holidays_finder.fit_transform(example_tsds)
    assert sorted(example_tsds.regressors) == sorted(expected_regressors)


def test_save_load(example_tsds):
    transform = HolidayTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)


def test_params_to_tune():
    transform = HolidayTransform()
    assert len(transform.params_to_tune()) == 0


@pytest.mark.parametrize(
    "freq, expected_result",
    (
        ("Q-JAN", [pd.Timestamp("1999-11-01 00:00:00"), pd.Timestamp("2000-01-31")]),
        ("M", [pd.Timestamp("2000-01-01 00:00:00"), pd.Timestamp("2000-01-31 00:00:00")]),
        ("W-MON", [pd.Timestamp("2000-01-31 00:00:00"), pd.Timestamp("2000-02-06 00:00:00")]),
    ),
)
def test_define_period_check_end(freq, expected_result):
    assert (define_period(pd.tseries.frequencies.to_offset(freq), pd.Timestamp("2000-01-31"), freq))[
        0
    ] == expected_result[0]
    assert (define_period(pd.tseries.frequencies.to_offset(freq), pd.Timestamp("2000-01-31"), freq))[
        1
    ] == expected_result[1]


@pytest.mark.parametrize(
    "freq, expected_result",
    (
        ("QS", [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-03-31")]),
        ("MS", [pd.Timestamp("2000-01-01 00:00:00"), pd.Timestamp("2000-01-31 00:00:00")]),
    ),
)
def test_define_period_check_start(freq, expected_result):
    assert (define_period(pd.tseries.frequencies.to_offset(freq), pd.Timestamp("2000-01-01"), freq))[
        0
    ] == expected_result[0]
    assert (define_period(pd.tseries.frequencies.to_offset(freq), pd.Timestamp("2000-01-01"), freq))[
        1
    ] == expected_result[1]


@pytest.mark.parametrize(
    "freq, expected_result",
    (
        ("Y", [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")]),
        ("YS", [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")]),
    ),
)
def test_define_period_year(freq, expected_result):
    assert (define_period(pd.tseries.frequencies.to_offset(freq), pd.Timestamp("2000-03-01"), freq))[
        0
    ] == expected_result[0]
    assert (define_period(pd.tseries.frequencies.to_offset(freq), pd.Timestamp("2000-03-01"), freq))[
        1
    ] == expected_result[1]
