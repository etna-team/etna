import numpy as np
import pandas as pd
import pytest

from etna.analysis.outliers import get_anomalies_mad
from etna.analysis.outliers.rolling_statistics import _sliding_window
from etna.analysis.outliers.rolling_statistics import _stl_decompose
from etna.analysis.outliers.rolling_statistics import sliding_window_decorator


@pytest.mark.parametrize(
    "window_size, stride, error",
    ((0, 1, "Window size must be positive integer!"), (2, 0, "Stride must be integer greater or equal to 1!")),
)
def test_sliding_window_params(window_size, stride, error):
    with pytest.raises(ValueError, match=error):
        _ = _sliding_window(x=np.empty, window_size=window_size, stride=stride)


@pytest.mark.parametrize("data", (np.arange(9),))
@pytest.mark.parametrize(
    "window_size, stride, answer",
    (
        (3, 1, np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])),
        (2, 1, np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])),
        (4, 1, np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]])),
        (3, 3, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        (7, 2, np.array([[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8]])),
        (1, 1, np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])),
        (1, 9, np.array([[8]])),
        (9, 1, np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])),
    ),
)
def test_sliding_window(data, window_size, stride, answer):
    res = _sliding_window(x=data, window_size=window_size, stride=stride)
    np.testing.assert_allclose(res, answer)


@pytest.mark.parametrize("data", (np.arange(9),))
@pytest.mark.parametrize(
    "window_size, stride, answer",
    (
        (3, 1, np.array([3, 6, 9, 12, 15, 18, 21])),
        (2, 1, np.array([1, 3, 5, 7, 9, 11, 13, 15])),
        (4, 1, np.array([6, 10, 14, 18, 22, 26])),
        (3, 3, np.array([3, 12, 21])),
        (7, 2, np.array([21, 35])),
        (1, 1, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
        (1, 9, np.array([8])),
        (9, 1, np.array([36])),
    ),
)
def test_sliding_window_decorator(data, window_size, stride, answer):
    apply_func = sliding_window_decorator(lambda x, idxs: np.sum(x[idxs]))
    res = apply_func(data, window_size=window_size, stride=stride, return_indices=False)
    np.testing.assert_allclose(res, answer)


def test_stl_decompose_error():
    with pytest.raises(ValueError, match="At least one component must be set!"):
        _ = _stl_decompose(series=pd.Series([0.0]), trend=False, seasonality=False)


def test_const_ts(const_ts_anomal):
    anomal = get_anomalies_mad(const_ts_anomal)
    assert len(anomal) == 0


def test_mad_outliers_missing_values(outliers_tsds):
    with pytest.raises(ValueError, match=".* contains missing values!"):
        _ = get_anomalies_mad(ts=outliers_tsds)


def test_mad_outliers_invalid_scale(outliers_df_with_two_columns):
    with pytest.raises(ValueError, match="Scaling parameter must be positive!"):
        _ = get_anomalies_mad(ts=outliers_df_with_two_columns, mad_scale=-1.0)


@pytest.mark.parametrize(
    "ts_name, error",
    (
        (
            "outliers_df_with_two_columns_int_timestamp",
            "Series must have inferable frequency to autodetect period for STL!",
        ),
        ("outliers_df_with_two_columns_minute_freq", "freq T not understood. Please report"),
    ),
)
def test_mad_outliers_stl_period_error(ts_name, error, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match=error):
        _ = get_anomalies_mad(ts=ts, seasonality=True)


@pytest.mark.parametrize(
    "ts_name, answer",
    (
        (
            "outliers_df_with_two_columns",
            {
                "1": [np.datetime64("2021-01-11")],
                "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-16"), np.datetime64("2021-01-27")],
            },
        ),
        ("outliers_df_with_two_columns_int_timestamp", {"1": [10], "2": [8, 15, 26]}),
        (
            "outliers_df_with_two_columns_minute_freq",
            {
                "1": [np.datetime64("2021-01-01T00:10:00")],
                "2": [
                    np.datetime64("2021-01-01T00:08:00"),
                    np.datetime64("2021-01-01T00:15:00"),
                    np.datetime64("2021-01-01T00:26:00"),
                ],
            },
        ),
    ),
)
def test_mad_outliers_various_index(ts_name, answer, request):
    ts = request.getfixturevalue(ts_name)
    res = get_anomalies_mad(ts=ts, window_size=30, stride=1, mad_scale=3)
    assert res == answer


@pytest.mark.parametrize(
    "window_size, mad_scale, stride, right_anomal",
    (
        (
            20,
            3,
            1,
            {
                "1": [np.datetime64("2021-01-11")],
                "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-16"), np.datetime64("2021-01-27")],
            },
        ),
        (
            20,
            7,
            1,
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
        (
            10,
            6,
            1,
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
        (
            12,
            4,
            3,
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
    ),
)
def test_mad_outliers(window_size, mad_scale, stride, right_anomal, outliers_df_with_two_columns):
    res = get_anomalies_mad(
        ts=outliers_df_with_two_columns, window_size=window_size, stride=stride, mad_scale=mad_scale
    )
    assert res == right_anomal


@pytest.mark.parametrize(
    "window_size, mad_scale, stride, right_anomal",
    (
        (
            15,
            3.2,
            1,
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
    ),
)
@pytest.mark.parametrize("period", (None, 2, 5))
def test_mad_outliers_with_seasonality(
    window_size, mad_scale, stride, period, right_anomal, outliers_df_with_two_columns
):
    res = get_anomalies_mad(
        ts=outliers_df_with_two_columns,
        window_size=window_size,
        stride=stride,
        mad_scale=mad_scale,
        period=period,
        seasonality=True,
    )
    assert len(res) == len(right_anomal)


@pytest.mark.parametrize(
    "window_size, mad_scale, stride, right_anomal",
    (
        (
            20,
            7,
            1,
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
    ),
)
@pytest.mark.parametrize("period", (None, 5))
def test_mad_outliers_with_trend(window_size, mad_scale, stride, period, right_anomal, outliers_df_with_two_columns):
    res = get_anomalies_mad(
        ts=outliers_df_with_two_columns,
        window_size=window_size,
        stride=stride,
        mad_scale=mad_scale,
        period=period,
        trend=True,
    )
    assert res == right_anomal


@pytest.mark.parametrize("true_params", (["1", "2"],))
@pytest.mark.parametrize("index_only", (True, False))
def test_interface_correct_args(true_params, index_only, outliers_df_with_two_columns):
    res = get_anomalies_mad(ts=outliers_df_with_two_columns, index_only=index_only)

    assert isinstance(res, dict)
    assert sorted(res.keys()) == sorted(true_params)

    for key in res:
        if index_only:
            assert isinstance(res[key], list)
            for value in res[key]:
                assert isinstance(value, np.datetime64)
        else:
            assert isinstance(res[key], pd.Series)


def test_in_column(outliers_df_with_two_columns):
    outliers = get_anomalies_mad(ts=outliers_df_with_two_columns, in_column="feature")
    expected = {"1": [np.datetime64("2021-01-08")], "2": [np.datetime64("2021-01-26")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])
