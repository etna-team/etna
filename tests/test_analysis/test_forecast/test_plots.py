import pandas as pd
import pytest

from etna.analysis import plot_residuals
from etna.analysis.forecast.plots import _get_borders_comparator
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform


def test_plot_residuals_fails_unkown_feature(example_tsdf):
    """Test that plot_residuals fails if meet unknown feature."""
    pipeline = Pipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=[5, 6, 7])], horizon=5
    )
    metrics, forecast_df, info = pipeline.backtest(ts=example_tsdf, metrics=[MAE()], n_folds=3)
    with pytest.raises(ValueError, match="Given feature isn't present in the dataset"):
        plot_residuals(forecast_df=forecast_df, ts=example_tsdf, feature="unkown_feature")


@pytest.mark.parametrize(
    "segments_df",
    (
        pd.DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]}),
        pd.DataFrame({"a": [0, 1, 2], "b": [-1, 0, 3]}),
        pd.DataFrame({"a": [0, 1, 2], "b": [-1, 2, 3]}),
        pd.DataFrame({"a": [0, 1, 2], "b": [-1, 3, 1]}),
        pd.DataFrame({"a": [0, 1, 2], "b": [-1, 1, 3]}),
        pd.DataFrame({"a": [0, 1, 2], "b": [1, 1, -3]}),
        pd.DataFrame({"a": [0, 1, 2], "b": [3, 2, 1]}),
    ),
)
def test_compare_error(segments_df):
    comparator = _get_borders_comparator(segment_borders=segments_df)
    with pytest.raises(ValueError, match="Detected intersection"):
        _ = comparator(name_a="a", name_b="b")


@pytest.mark.parametrize(
    "segments_df,expected",
    (
        (pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]}), 0),
        (pd.DataFrame({"a": [0, 1, 2], "b": [-2, -1, 0]}), 1),
        (pd.DataFrame({"a": [0, 1, 2], "b": [-1, -2, -3]}), 1),
        (pd.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]}), -1),
        (pd.DataFrame({"a": [0, 1, 2], "b": [3, 2, 3]}), -1),
    ),
)
def test_compare(segments_df, expected):
    comparator = _get_borders_comparator(segment_borders=segments_df)
    assert comparator(name_a="a", name_b="b") == expected
