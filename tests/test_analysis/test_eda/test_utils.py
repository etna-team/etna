import numpy as np
import pandas as pd
import pytest

from etna.analysis.eda.plots import _create_holidays_df
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from tests.utils import convert_ts_to_int_timestamp


@pytest.fixture
def simple_tsdf_int_timestamp(simple_tsdf) -> TSDataset:
    return convert_ts_to_int_timestamp(ts=simple_tsdf)


def test_create_holidays_df_str_fail_as_is(simple_tsdf):
    with pytest.raises(ValueError, match="Parameter `as_is` should be used with"):
        _create_holidays_df("RU", simple_tsdf.index, as_is=True)


def test_create_holidays_df_str_fail_int_timestamp(simple_tsdf_int_timestamp):
    with pytest.raises(ValueError, match="Parameter `holidays` should be pd.DataFrame for data with integer timestamp"):
        _create_holidays_df("RU", simple_tsdf_int_timestamp.index, as_is=False)


def test_create_holidays_df_str_non_existing_country(simple_tsdf):
    with pytest.raises((NotImplementedError, KeyError)):
        _create_holidays_df("THIS_COUNTRY_DOES_NOT_EXIST", simple_tsdf.index, as_is=False)


def test_create_holidays_df_str(simple_tsdf):
    df = _create_holidays_df("RU", simple_tsdf.index, as_is=False)
    assert len(df) == len(simple_tsdf.df)
    assert all(df.dtypes == bool)


def test_create_holidays_df_empty_fail(simple_tsdf):
    with pytest.raises(ValueError):
        _create_holidays_df(pd.DataFrame(), simple_tsdf.index, as_is=False)


def test_create_holidays_df_intersect_none(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["1900-01-01", "1901-01-01"])})
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert not df.all(axis=None)


def test_create_holidays_df_one_day(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01"])})
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 1
    assert "New Year" in df.columns


def test_create_holidays_df_upper_window(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01"]), "upper_window": 2})
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 3


def test_create_holidays_df_upper_window_out_of_index(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2019-12-25"]), "upper_window": 10})
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 4


def test_create_holidays_df_lower_window(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": -2})
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 3


def test_create_holidays_df_lower_window_out_of_index(simple_tsdf):
    holidays = pd.DataFrame(
        {"holiday": "Moscow Anime Festival", "ds": pd.to_datetime(["2020-02-22"]), "lower_window": -5}
    )
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 2


def test_create_holidays_df_lower_upper_windows(simple_tsdf):
    holidays = pd.DataFrame(
        {"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "upper_window": 3, "lower_window": -3}
    )
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 7


def test_create_holidays_df_as_is(simple_tsdf):
    holidays = pd.DataFrame(index=pd.date_range(start="2020-01-07", end="2020-01-10"), columns=["Christmas"], data=1)
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=True)
    assert df.sum().sum() == 4


def test_create_holidays_df_as_is_int_timestamp(simple_tsdf_int_timestamp):
    holidays = pd.DataFrame(index=np.arange(7, 11), columns=["Christmas"], data=1)
    df = _create_holidays_df(holidays, simple_tsdf_int_timestamp.index, as_is=True)
    assert df.sum().sum() == 4


def test_create_holidays_df_h_freq():
    classic_df = generate_ar_df(periods=30, start_time="2020-01-01", n_segments=1, freq="H")
    ts = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame(
        {
            "holiday": "Christmas",
            "ds": pd.to_datetime(
                ["2020-01-01"],
            ),
            "upper_window": 3,
        }
    )
    df = _create_holidays_df(holidays, ts.index, as_is=False)
    assert df.sum().sum() == 4


def test_create_holidays_df_15t_freq():
    classic_df = generate_ar_df(periods=30, start_time="2020-01-01", n_segments=1, freq="15T")
    ts = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01 01:00:00"]), "upper_window": 3})
    df = _create_holidays_df(holidays, ts.index, as_is=False)
    assert df.sum().sum() == 4
    assert df.loc["2020-01-01 01:00:00":"2020-01-01 01:45:00"].sum().sum() == 4


def test_create_holidays_df_int_freq():
    classic_df = generate_ar_df(periods=30, start_time=0, n_segments=1, freq=None)
    ts = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame(
        {
            "holiday": "Christmas",
            "ds": [3],
            "upper_window": 3,
        }
    )
    df = _create_holidays_df(holidays, ts.index, as_is=False)
    assert df.sum().sum() == 4


def test_create_holidays_df_several_holidays(simple_tsdf):
    christmas = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": -3})
    new_year = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01"]), "upper_window": 2})
    holidays = pd.concat((christmas, new_year))
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 7


def test_create_holidays_df_zero_windows(simple_tsdf):
    holidays = pd.DataFrame(
        {"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": 0, "upper_window": 0}
    )
    df = _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
    assert df.sum().sum() == 1
    assert df.loc["2020-01-07"].sum() == 1


def test_create_holidays_df_upper_window_negative(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "upper_window": -1})
    with pytest.raises(ValueError):
        _create_holidays_df(holidays, simple_tsdf.index, as_is=False)


def test_create_holidays_df_lower_window_positive(simple_tsdf):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": 1})
    with pytest.raises(ValueError):
        _create_holidays_df(holidays, simple_tsdf.index, as_is=False)
