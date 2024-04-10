import pytest

from etna.analysis import get_anomalies_density
from etna.analysis import plot_anomalies
from etna.analysis import plot_anomalies_interactive


@pytest.mark.parametrize(
    "ts_name, params, match",
    [
        ("example_tsdf", {"start": 10}, "Parameter start has incorrect type"),
        ("example_tsdf", {"end": 10}, "Parameter end has incorrect type"),
        ("example_tsdf_int_timestamp", {"start": "2020-01-01"}, "Parameter start has incorrect type"),
        ("example_tsdf_int_timestamp", {"end": "2020-01-01"}, "Parameter end has incorrect type"),
    ],
)
def test_plot_anomalies_fail_incorrect_start_end_type(ts_name, params, match, request):
    ts = request.getfixturevalue(ts_name)
    anomaly_dict = {"segment_1": [10, 100], "segment_2": [20, 200]}
    with pytest.raises(ValueError, match=match):
        plot_anomalies(ts=ts, anomaly_dict=anomaly_dict, **params)


@pytest.mark.parametrize(
    "ts_name, params, match",
    [
        ("example_tsdf", {"start": 10}, "Parameter start has incorrect type"),
        ("example_tsdf", {"end": 10}, "Parameter end has incorrect type"),
        ("example_tsdf_int_timestamp", {"start": "2020-01-01"}, "Parameter start has incorrect type"),
        ("example_tsdf_int_timestamp", {"end": "2020-01-01"}, "Parameter end has incorrect type"),
    ],
)
def test_plot_anomalies_interactive_fail_incorrect_start_end_type(ts_name, params, match, request):
    ts = request.getfixturevalue(ts_name)
    params_bounds = {"window_size": (5, 20, 1), "distance_coef": (0.1, 3, 0.25)}
    with pytest.raises(ValueError, match=match):
        plot_anomalies_interactive(
            ts=ts,
            segment="segment_1",
            method=get_anomalies_density,
            params_bounds=params_bounds,
            **params,
        )
