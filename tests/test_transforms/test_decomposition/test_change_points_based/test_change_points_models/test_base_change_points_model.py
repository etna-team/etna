import pandas as pd
import pytest

from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter


@pytest.mark.parametrize(
    "change_points, expected_intervals",
    [
        (
            [],
            [
                (None, None),
            ],
        ),
        (
            [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18"), pd.Timestamp("2020-02-24")],
            [
                (None, pd.Timestamp("2020-01-01")),
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18")),
                (pd.Timestamp("2020-01-18"), pd.Timestamp("2020-02-24")),
                (pd.Timestamp("2020-02-24"), None),
            ],
        ),
        (
            [10, 20, 30],
            [
                (None, 10),
                (10, 20),
                (20, 30),
                (30, None),
            ],
        ),
    ],
)
def test_build_intervals(change_points, expected_intervals):
    """Check correctness of intervals generation with list of change points."""
    intervals = BaseChangePointsModelAdapter._build_intervals(change_points=change_points)
    assert isinstance(intervals, list)
    assert len(intervals) == len(expected_intervals)
    for (exp_left, exp_right), (real_left, real_right) in zip(expected_intervals, intervals):
        assert exp_left == real_left
        assert exp_right == real_right
