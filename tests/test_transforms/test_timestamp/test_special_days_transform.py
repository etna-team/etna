from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.timestamp import SpecialDaysTransform
from etna.transforms.timestamp.special_days import _OneSegmentSpecialDaysTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import convert_ts_to_int_timestamp


@pytest.fixture()
def constant_days_df():
    """Create pandas dataframe that represents one segment and has const value column"""
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-01", end="2020-04-01", freq=pd.offsets.Day())})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def df_with_specials():
    """Create pandas dataframe that represents one segment and has non-const value column."""
    weekday_outliers_dates = [
        # monday
        {"timestamp": datetime(2020, 12, 28).date(), "target": 10},
        # tuesday
        {"timestamp": datetime(2020, 1, 7).date(), "target": 20},
        # wednesdays
        {"timestamp": datetime(2020, 2, 12).date(), "target": 5},
        {"timestamp": datetime(2020, 9, 30).date(), "target": 10},
        {"timestamp": datetime(2020, 6, 10).date(), "target": 14},
        # sunday
        {"timestamp": datetime(2020, 5, 10).date(), "target": 12},
    ]
    special_df = pd.DataFrame(weekday_outliers_dates)
    special_df["timestamp"] = pd.to_datetime(special_df["timestamp"])
    date_range = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-12-31")})
    df = pd.merge(date_range, special_df, on="timestamp", how="left").fillna(0)

    special_weekdays = (2,)
    special_monthdays = (7, 10)

    df["week_true"] = df["timestamp"].apply(lambda x: x.weekday() in special_weekdays).astype(int).astype("category")
    df["month_true"] = df["timestamp"].apply(lambda x: x.day in special_monthdays).astype(int).astype("category")
    df["external_timestamp"] = df["timestamp"]
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def ts_with_specials(df_with_specials):
    """Create dataset with special weekdays and monthdays."""
    flat_df = df_with_specials.reset_index()
    flat_df["segment"] = "1"
    flat_df = flat_df[["timestamp", "segment", "target"]]

    wide_df = TSDataset.to_dataset(flat_df)

    flat_df["external_timestamp"] = flat_df["timestamp"]
    flat_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(flat_df)

    ts = TSDataset(df=wide_df, df_exog=df_exog, freq=pd.offsets.Day())
    return ts


@pytest.fixture()
def ts_with_specials_and_regressor(ts_with_specials) -> TSDataset:
    df = ts_with_specials._raw_df
    df_exog = ts_with_specials._df_exog
    ts = TSDataset(df=df.iloc[:-10], df_exog=df_exog, freq=ts_with_specials.freq, known_future=["external_timestamp"])
    return ts


@pytest.fixture()
def ts_with_specials_and_nans_in_timestamp(ts_with_specials) -> TSDataset:
    ts = ts_with_specials
    df = ts._raw_df
    df_exog = ts._df_exog
    df_exog.loc[df_exog.index[:3], pd.IndexSlice[:, "external_timestamp"]] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture()
def constant_days_two_segments_ts(constant_days_df: pd.DataFrame):
    """Create TSDataset that has two segments with constant columns each."""
    df_1 = constant_days_df.reset_index()
    df_2 = constant_days_df.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=df, freq=pd.offsets.Day())
    return ts


def test_interface_week(constant_days_df: pd.DataFrame):
    """
    This test checks that _OneSegmentSpecialDaysTransform that should find special weekdays creates the only column with
    'anomaly_weekdays' name as expected.
    """
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=True, find_special_month_day=False)
    df = special_days_finder.fit_transform(constant_days_df)
    assert "anomaly_weekdays" in df.columns
    assert "anomaly_monthdays" not in df.columns
    assert df["anomaly_weekdays"].dtype == "category"


def test_interface_month(constant_days_df: pd.DataFrame):
    """
    This test checks that _OneSegmentSpecialDaysTransform that should find special month days creates the only column with
    'anomaly_monthdays' name as expected.
    """
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=False, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_df)
    assert "anomaly_weekdays" not in df.columns
    assert "anomaly_monthdays" in df.columns
    assert df["anomaly_monthdays"].dtype == "category"


def test_interface_week_month(constant_days_df: pd.DataFrame):
    """
    This test checks that _OneSegmentSpecialDaysTransform that should find special month and week days
    creates two columns with 'anomaly_monthdays' and 'anomaly_weekdays' name as expected.
    """
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_df)
    assert "anomaly_weekdays" in df.columns
    assert "anomaly_monthdays" in df.columns
    assert df["anomaly_weekdays"].dtype == "category"
    assert df["anomaly_monthdays"].dtype == "category"


def test_interface_noweek_nomonth():
    """This test checks that bad-inited _OneSegmentSpecialDaysTransform raises AssertionError."""
    with pytest.raises(ValueError):
        _ = _OneSegmentSpecialDaysTransform(find_special_weekday=False, find_special_month_day=False)


def test_interface_two_segments_week(constant_days_two_segments_ts: TSDataset):
    """
    This test checks that SpecialDaysTransform that should find special weekdays creates the only column with
    'anomaly_weekdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=False)
    df = special_days_finder.fit_transform(constant_days_two_segments_ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" in df[segment].columns
        assert "anomaly_monthdays" not in df[segment].columns
        assert df[segment]["anomaly_weekdays"].dtype == "category"


def test_interface_two_segments_month(constant_days_two_segments_ts: TSDataset):
    """
    This test checks that SpecialDaysTransform that should find special month days creates the only column with
    'anomaly_monthdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=False, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_two_segments_ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" not in df[segment].columns
        assert "anomaly_monthdays" in df[segment].columns
        assert df[segment]["anomaly_monthdays"].dtype == "category"


def test_interface_two_segments_week_month(constant_days_two_segments_ts: TSDataset):
    """
    This test checks that SpecialDaysTransform that should find special month and week days
    creates two columns with 'anomaly_monthdays' and 'anomaly_weekdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_two_segments_ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" in df[segment].columns
        assert "anomaly_monthdays" in df[segment].columns
        assert df[segment]["anomaly_weekdays"].dtype == "category"
        assert df[segment]["anomaly_monthdays"].dtype == "category"


def test_interface_two_segments_noweek_nomonth():
    """This test checks that bad-inited SpecialDaysTransform raises ValueError on initialisation."""
    with pytest.raises(
        ValueError, match="_OneSegmentSpecialDaysTransform feature does nothing with given init args configuration"
    ):
        _ = SpecialDaysTransform(find_special_weekday=False, find_special_month_day=False)


@pytest.mark.parametrize("in_column", [None, "external_timestamp"])
def test_week_feature(in_column: Optional[str], df_with_specials: pd.DataFrame):
    """This test checks that _OneSegmentSpecialDaysTransform computes weekday feature correctly."""
    special_days_finder = _OneSegmentSpecialDaysTransform(
        find_special_weekday=True, find_special_month_day=False, in_column=in_column
    )
    df = special_days_finder.fit_transform(df_with_specials)
    pd.testing.assert_series_equal(df_with_specials["week_true"], df["anomaly_weekdays"], check_names=False)


@pytest.mark.parametrize("in_column", [None, "external_timestamp"])
def test_month_feature(in_column: Optional[str], df_with_specials: pd.DataFrame):
    """This test checks that _OneSegmentSpecialDaysTransform computes monthday feature correctly."""
    special_days_finder = _OneSegmentSpecialDaysTransform(
        find_special_weekday=False, find_special_month_day=True, in_column=in_column
    )
    df = special_days_finder.fit_transform(df_with_specials)
    pd.testing.assert_series_equal(df_with_specials["month_true"], df["anomaly_monthdays"], check_names=False)


def test_no_false_positive_week(constant_days_df: pd.DataFrame):
    """This test checks that there is no false-positive results in week mode."""
    special_days_finder = _OneSegmentSpecialDaysTransform()
    res = special_days_finder.fit_transform(constant_days_df)
    assert res["anomaly_weekdays"].astype("bool").sum() == 0


def test_no_false_positive_month(constant_days_df: pd.DataFrame):
    """This test checks that there is no false-positive results in month mode."""
    special_days_finder = _OneSegmentSpecialDaysTransform()
    res = special_days_finder.fit_transform(constant_days_df)
    assert res["anomaly_monthdays"].astype("bool").sum() == 0


def test_transform_raise_error_if_not_fitted(constant_days_df: pd.DataFrame):
    """Test that transform for one segment raise error when calling transform without being fit."""
    transform = _OneSegmentSpecialDaysTransform()
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=constant_days_df)


def test_fit_transform_with_nans_in_target(ts_diff_endings):
    transform = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    transform.fit_transform(ts_diff_endings)


def test_fit_transform_with_nans_in_timestamp(ts_with_specials_and_nans_in_timestamp):
    ts = ts_with_specials_and_nans_in_timestamp
    transform = SpecialDaysTransform(
        find_special_weekday=True, find_special_month_day=True, in_column="external_timestamp"
    )
    result = transform.fit_transform(ts)
    columns = ["anomaly_weekdays", "anomaly_monthdays"]
    for segment in ts.segments:
        result_df = result._df.loc[:, pd.IndexSlice[segment, columns]]
        assert np.all(result_df.isna().sum() == 3)


def test_transform_index_fail_int_timestamp(ts_with_specials):
    ts = convert_ts_to_int_timestamp(ts=ts_with_specials)
    transform = SpecialDaysTransform(in_column=None)
    with pytest.raises(ValueError, match="Transform can't work with integer index, parameter in_column should be set"):
        _ = transform.fit(ts)


def test_get_regressors_info_index(ts_with_specials):
    transform = SpecialDaysTransform(in_column=None)

    regressors_info = transform.get_regressors_info()

    expected_regressor_info = ["anomaly_weekdays", "anomaly_monthdays"]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


def test_get_regressors_info_in_column_fail_not_fitted(ts_with_specials):
    transform = SpecialDaysTransform(in_column="external_timestamp")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


def test_get_regressors_info_in_column_fitted_exog(ts_with_specials):
    transform = SpecialDaysTransform(in_column="external_timestamp")

    transform.fit(ts_with_specials)
    regressors_info = transform.get_regressors_info()

    assert regressors_info == []


def test_get_regressors_info_in_column_fitted_regressor(ts_with_specials_and_regressor):
    transform = SpecialDaysTransform(in_column="external_timestamp")

    transform.fit(ts_with_specials_and_regressor)
    regressors_info = transform.get_regressors_info()

    expected_regressor_info = ["anomaly_weekdays", "anomaly_monthdays"]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


def test_save_load(ts_with_specials):
    ts = ts_with_specials
    transform = SpecialDaysTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


def test_params_to_tune(ts_with_specials):
    def skip_parameters(parameters):
        names = ["find_special_weekday", "find_special_month_day"]
        values = [not parameters[x] for x in names]
        if all(values):
            return True
        return False

    transform = SpecialDaysTransform()
    ts = ts_with_specials
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts, skip_parameters=skip_parameters)
