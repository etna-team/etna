import pytest
from ruptures.detection import Binseg
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor

from etna.analysis import plot_change_points_interactive
from etna.analysis import plot_time_series_with_change_points
from etna.analysis import plot_trend
from etna.analysis import seasonal_plot
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import STLTransform
from etna.transforms import TheilSenTrendTransform


@pytest.mark.parametrize(
    "poly_degree, trend_transform_class",
    (
        [1, LinearTrendTransform],
        [2, LinearTrendTransform],
        [1, TheilSenTrendTransform],
        [2, TheilSenTrendTransform],
    ),
)
def test_plot_trend(poly_degree, example_tsdf, trend_transform_class):
    plot_trend(ts=example_tsdf, trend_transform=trend_transform_class(in_column="target", poly_degree=poly_degree))


@pytest.mark.parametrize("detrend_model", (TheilSenRegressor(), LinearRegression()))
def test_plot_bin_seg(example_tsdf, detrend_model):
    plot_trend(ts=example_tsdf, trend_transform=ChangePointsTrendTransform(in_column="target"))


@pytest.mark.parametrize("period", (7, 30))
def test_plot_stl(example_tsdf, period):
    plot_trend(ts=example_tsdf, trend_transform=STLTransform(in_column="target", period=period))


@pytest.mark.parametrize(
    "freq, cycle, additional_params",
    [
        ("D", 5, dict(alignment="first")),
        ("D", 5, dict(alignment="last")),
        ("D", "week", {}),
        ("D", "month", {}),
        ("D", "year", {}),
        ("M", "year", dict(aggregation="sum")),
        ("M", "year", dict(aggregation="mean")),
    ],
)
def test_dummy_seasonal_plot(freq, cycle, additional_params, ts_with_different_series_length):
    seasonal_plot(ts=ts_with_different_series_length, freq=freq, cycle=cycle, **additional_params)


@pytest.mark.parametrize(
    "ts_name, params, match",
    [
        ("example_tsdf", {"start": 10}, "Parameter start has incorrect type"),
        ("example_tsdf", {"end": 10}, "Parameter end has incorrect type"),
        ("example_tsdf_int_timestamp", {"start": "2020-01-01"}, "Parameter start has incorrect type"),
        ("example_tsdf_int_timestamp", {"end": "2020-01-01"}, "Parameter end has incorrect type"),
    ],
)
def test_plot_time_series_with_change_points_fail_incorrect_start_end_type(ts_name, params, match, request):
    ts = request.getfixturevalue(ts_name)
    change_points = {"segment_1": [10, 100], "segment_2": [20, 200]}
    with pytest.raises(ValueError, match=match):
        plot_time_series_with_change_points(ts=ts, change_points=change_points, **params)


@pytest.mark.parametrize(
    "ts_name, params, match",
    [
        ("example_tsdf", {"start": 10}, "Parameter start has incorrect type"),
        ("example_tsdf", {"end": 10}, "Parameter end has incorrect type"),
        ("example_tsdf_int_timestamp", {"start": "2020-01-01"}, "Parameter start has incorrect type"),
        ("example_tsdf_int_timestamp", {"end": "2020-01-01"}, "Parameter end has incorrect type"),
    ],
)
def test_plot_change_points_interactive_fail_incorrect_start_end_type(ts_name, params, match, request):
    ts = request.getfixturevalue(ts_name)
    params_bounds = {"n_bkps": [0, 5, 1], "min_size": [1, 10, 3]}
    with pytest.raises(ValueError, match=match):
        plot_change_points_interactive(
            ts=ts,
            change_point_model=Binseg,
            model="l2",
            params_bounds=params_bounds,
            model_params=["min_size"],
            predict_params=["n_bkps"],
            **params,
        )


@pytest.mark.parametrize("alignment", ["first", "last"])
def test_seasonal_plot_datetime_timestamp(alignment, example_tsdf):
    seasonal_plot(ts=example_tsdf, cycle=10, alignment=alignment)


@pytest.mark.parametrize("alignment", ["first", "last"])
def test_seasonal_plot_int_timestamp(alignment, example_tsdf_int_timestamp):
    seasonal_plot(ts=example_tsdf_int_timestamp, cycle=10, alignment=alignment)


def test_seasonal_plot_int_timestamp_fail_resample(example_tsdf_int_timestamp):
    with pytest.raises(ValueError, match="Resampling isn't supported for data with integer timestamp"):
        seasonal_plot(ts=example_tsdf_int_timestamp, freq="D", cycle=10)


def test_seasonal_plot_int_timestamp_fail_non_int_cycle(example_tsdf_int_timestamp):
    with pytest.raises(ValueError, match="Setting non-integer cycle isn't supported"):
        seasonal_plot(ts=example_tsdf_int_timestamp, freq=None, cycle="year")


def test_seasonal_plot_datetime_timestamp_fail_none_freq(example_tsdf):
    with pytest.raises(ValueError, match="Value None for freq parameter isn't supported"):
        seasonal_plot(ts=example_tsdf, freq=None, cycle=10)
