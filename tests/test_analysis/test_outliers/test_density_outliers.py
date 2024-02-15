from typing import Callable
from typing import List
from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from etna.analysis.outliers.density_outliers import absolute_difference_distance
from etna.analysis.outliers.density_outliers import get_anomalies_density
from etna.analysis.outliers.density_outliers import get_segment_density_outliers_indices
from etna.datasets.tsdataset import TSDataset


@pytest.fixture
def simple_window() -> np.array:
    return np.array([4, 5, 6, 4, 100, 200, 2])


def test_const_ts(const_ts_anomal):
    anomal = get_anomalies_density(const_ts_anomal)
    assert len(anomal) == 0


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (0, 0, 0),
        (2, 0, 2),
        (0, 2, 2),
        (-2, 0, 2),
        (0, -2, 2),
        (2, 2, 0),
        (5, 3, 2),
        (3, 5, 2),
        (5, -3, 8),
        (-3, 5, 8),
        (-5, -2, 3),
        (-2, -5, 3),
    ],
)
def test_default_distance(x, y, expected):
    assert absolute_difference_distance(x, y) == expected


def test_absolute_diff_out_format():
    out = absolute_difference_distance(np.ones(3), np.ones(3))
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize(
    "window_size,n_neighbors,distance_threshold,expected",
    (
        (5, 2, 2.5, [4, 5, 6]),
        (6, 3, 10, [4, 5]),
        (2, 1, 1.8, [3, 4, 5, 6]),
        (3, 1, 120, []),
        (100, 2, 1.5, [2, 4, 5, 6]),
    ),
)
@pytest.mark.parametrize("distance_func", ("absolute_difference", lambda x, y: abs(x - y)))
def test_get_segment_density_outliers_indices(
    simple_window: np.array,
    window_size: int,
    n_neighbors: int,
    distance_threshold: float,
    expected: List[int],
    distance_func: Union[str, Callable[[float, float], float]],
):
    """Check that outliers in one series computation works correctly."""
    outliers = get_segment_density_outliers_indices(
        series=simple_window, window_size=window_size, n_neighbors=n_neighbors, distance_threshold=distance_threshold
    )
    np.testing.assert_array_equal(outliers, expected)


@pytest.mark.parametrize("distance_func", ("abc",))
def test_get_anomalies_density_invalid_distance(outliers_tsds: TSDataset, distance_func: str):
    with pytest.raises(NotImplementedError, match=".* is not a valid DistanceFunction"):
        _ = get_anomalies_density(
            ts=outliers_tsds, window_size=7, distance_coef=2, n_neighbors=3, distance_func=distance_func
        )


def test_get_anomalies_density_interface(outliers_tsds: TSDataset):
    outliers = get_anomalies_density(ts=outliers_tsds, window_size=7, distance_coef=2, n_neighbors=3)
    for segment in ["1", "2"]:
        assert segment in outliers
        assert isinstance(outliers[segment], list)


def test_get_anomalies_density(outliers_tsds: TSDataset):
    """Check if get_anomalies_density works correctly."""
    outliers = get_anomalies_density(ts=outliers_tsds, window_size=7, distance_coef=2.1, n_neighbors=3)
    expected = {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])


def test_get_anomalies_density_custom_func_called(outliers_tsds: TSDataset):
    mock = MagicMock(return_value=0.0)
    _ = get_anomalies_density(ts=outliers_tsds, window_size=7, distance_coef=2.1, n_neighbors=3, distance_func=mock)
    mock.assert_called()


@pytest.mark.parametrize("index_only, value_type", ((True, list), (False, pd.Series)))
def test_get_anomalies_density_index_only(outliers_tsds: TSDataset, index_only: bool, value_type):
    result = get_anomalies_density(
        ts=outliers_tsds, window_size=7, distance_coef=2.1, n_neighbors=3, index_only=index_only
    )

    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, value_type)


def test_in_column(outliers_df_with_two_columns):
    outliers = get_anomalies_density(ts=outliers_df_with_two_columns, in_column="feature", window_size=10)
    expected = {"1": [np.datetime64("2021-01-08")], "2": [np.datetime64("2021-01-26")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])
