import numpy as np
import pandas as pd
import pytest

from etna.analysis.decomposition.utils import _get_labels_names
from etna.analysis.decomposition.utils import _resample
from etna.analysis.decomposition.utils import _seasonal_split
from etna.datasets import TSDataset
from etna.transforms import LinearTrendTransform
from etna.transforms import TheilSenTrendTransform


@pytest.mark.parametrize(
    "poly_degree, expect_values, trend_class",
    (
        [1, True, LinearTrendTransform],
        [2, False, LinearTrendTransform],
        [1, True, TheilSenTrendTransform],
        [2, False, TheilSenTrendTransform],
    ),
)
def test_get_labels_names_linear_coeffs(example_tsdf, poly_degree, expect_values, trend_class):
    ln_tr = trend_class(in_column="target", poly_degree=poly_degree)
    ln_tr.fit_transform(example_tsdf)
    segments = example_tsdf.segments
    _, linear_coeffs = _get_labels_names([ln_tr], segments)
    if expect_values:
        assert list(linear_coeffs.values()) != ["", ""]
    else:
        assert list(linear_coeffs.values()) == ["", ""]


@pytest.mark.parametrize(
    "timestamp, freq_offset, cycle, expected_cycle_names, expected_in_cycle_nums, expected_in_cycle_names",
    [
        (
            pd.date_range(start="2020-01-01", periods=5, freq=pd.offsets.Day()).to_series(),
            pd.offsets.Day(),
            3,
            ["1", "1", "1", "2", "2"],
            [0, 1, 2, 0, 1],
            ["0", "1", "2", "0", "1"],
        ),
        (
            pd.date_range(start="2020-01-01", periods=6, freq=pd.offsets.Minute(n=15)).to_series(),
            pd.offsets.Minute(n=15),
            "hour",
            ["2020-01-01 00"] * 4 + ["2020-01-01 01"] * 2,
            [0, 1, 2, 3, 0, 1],
            ["0", "1", "2", "3", "0", "1"],
        ),
        (
            pd.date_range(start="2020-01-01", periods=26, freq=pd.offsets.Hour()).to_series(),
            pd.offsets.Hour(),
            "day",
            ["2020-01-01"] * 24 + ["2020-01-02"] * 2,
            [i % 24 for i in range(26)],
            [str(i % 24) for i in range(26)],
        ),
        (
            pd.date_range(start="2020-01-01", periods=10, freq=pd.offsets.Day()).to_series(),
            pd.offsets.Day(),
            "week",
            ["2020-00"] * 5 + ["2020-01"] * 5,
            [2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
            ["Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        ),
        (
            pd.date_range(start="2020-01-03", periods=40, freq=pd.offsets.Day()).to_series(),
            pd.offsets.Day(),
            "month",
            ["2020-Jan"] * 29 + ["2020-Feb"] * 11,
            list(range(3, 32)) + list(range(1, 12)),
            [str(i) for i in range(3, 32)] + [str(i) for i in range(1, 12)],
        ),
        (
            pd.date_range(start="2020-01-01", periods=14, freq=pd.offsets.MonthEnd()).to_series(),
            pd.offsets.MonthEnd(),
            "quarter",
            ["2020-1"] * 3 + ["2020-2"] * 3 + ["2020-3"] * 3 + ["2020-4"] * 3 + ["2021-1"] * 2,
            [i % 3 for i in range(14)],
            [str(i % 3) for i in range(14)],
        ),
        (
            pd.date_range(start="2020-01-01", periods=14, freq=pd.offsets.MonthEnd()).to_series(),
            pd.offsets.MonthEnd(),
            "year",
            ["2020"] * 12 + ["2021"] * 2,
            [i % 12 + 1 for i in range(14)],
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"],
        ),
        (
            pd.Series(np.arange(5, 10)),
            None,
            3,
            ["1", "1", "1", "2", "2"],
            [0, 1, 2, 0, 1],
            ["0", "1", "2", "0", "1"],
        ),
    ],
)
def test_seasonal_split(
    timestamp, freq_offset, cycle, expected_cycle_names, expected_in_cycle_nums, expected_in_cycle_names
):
    cycle_df = _seasonal_split(timestamp=timestamp, freq_offset=freq_offset, cycle=cycle)
    assert cycle_df["cycle_name"].tolist() == expected_cycle_names
    assert cycle_df["in_cycle_num"].tolist() == expected_in_cycle_nums
    assert cycle_df["in_cycle_name"].tolist() == expected_in_cycle_names


@pytest.mark.parametrize(
    "timestamp, values, resample_freq_offset, aggregation, expected_timestamp, expected_values",
    [
        (
            pd.date_range(start="2020-01-01", periods=14, freq=pd.offsets.QuarterEnd()),
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 10, 16, 10, 5, 7, 5, 7, 3, 3],
            pd.offsets.YearEnd(),
            "sum",
            pd.date_range(start="2020-01-01", periods=4, freq=pd.offsets.YearEnd()),
            [np.NaN, 36.0, 24.0, 6.0],
        ),
        (
            pd.date_range(start="2020-01-01", periods=14, freq=pd.offsets.QuarterEnd()),
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 10, 16, 10, 5, 7, 5, 7, 3, 3],
            pd.offsets.YearEnd(),
            "mean",
            pd.date_range(start="2020-01-01", periods=4, freq=pd.offsets.YearEnd()),
            [np.NaN, 12.0, 6.0, 3.0],
        ),
        (
            pd.date_range(start="2020-01-01", periods=4, freq=pd.offsets.YearEnd()),
            [np.NaN, 12.0, 6.0, 3.0],
            pd.offsets.QuarterEnd(),
            "mean",
            pd.date_range(start="2020-01-01", periods=4, freq=pd.offsets.YearEnd()),
            [np.NaN, 12.0, 6.0, 3.0],
        ),
    ],
)
def test_resample(timestamp, values, resample_freq_offset, aggregation, expected_timestamp, expected_values):
    df = pd.DataFrame({"timestamp": timestamp.tolist(), "target": values, "segment": len(timestamp) * ["segment_0"]})
    df_wide = TSDataset.to_dataset(df)
    df_resampled = _resample(df=df_wide, freq_offset=resample_freq_offset, aggregation=aggregation)
    assert df_resampled.index.tolist() == expected_timestamp.tolist()
    assert (
        df_resampled.loc[:, pd.IndexSlice["segment_0", "target"]]
        .reset_index(drop=True)
        .equals(pd.Series(expected_values))
    )
