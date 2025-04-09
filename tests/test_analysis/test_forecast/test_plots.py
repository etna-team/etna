import pandas as pd
import pytest

from etna.analysis import metric_per_segment_distribution_plot
from etna.analysis import plot_metric_per_segment
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
    metrics, forecast_ts_list, info, pipelines = pipeline.backtest(ts=example_tsdf, metrics=[MAE()], n_folds=3).values()
    with pytest.raises(ValueError, match="Given feature isn't present in the dataset"):
        plot_residuals(forecast_ts_list=forecast_ts_list, ts=example_tsdf, feature="unkown_feature")


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


@pytest.fixture
def metrics_df_with_folds() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "segment": ["segment_0"] * 3 + ["segment_1"] * 3 + ["segment_2"] * 3,
            "MAE": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            "MSE": [None, 3.0, 4.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            "MAPE": [None, None, None, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0],
            "SMAPE": [None, None, None, None, None, None, 50.0, 60.0, 70.0],
            "RMSE": [None, None, None, None, None, None, None, None, None],
            "fold_number": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    return df


@pytest.fixture
def metrics_df_no_folds(metrics_df_with_folds) -> pd.DataFrame:
    df = metrics_df_with_folds
    df = df.groupby("segment").mean(numeric_only=False).reset_index().drop("fold_number", axis=1)
    return df


@pytest.mark.parametrize(
    "df_name, metric_name",
    [
        ("metrics_df_with_folds", "MAE"),
        ("metrics_df_no_folds", "MAE"),
        ("metrics_df_no_folds", "MSE"),
    ],
)
def test_plot_metric_per_segment_ok(df_name, metric_name, request):
    metrics_df = request.getfixturevalue(df_name)
    plot_metric_per_segment(metrics_df=metrics_df, metric_name=metric_name)


@pytest.mark.parametrize(
    "df_name, metric_name",
    [
        ("metrics_df_with_folds", "MAPE"),
        ("metrics_df_no_folds", "RMSE"),
    ],
)
def test_plot_metric_per_segment_warning_empty_segments(df_name, metric_name, request):
    metrics_df = request.getfixturevalue(df_name)
    with pytest.warns(UserWarning, match="There are segments with all missing metric values"):
        plot_metric_per_segment(metrics_df=metrics_df, metric_name=metric_name)


@pytest.mark.parametrize(
    "df_name, metric_name",
    [
        ("metrics_df_with_folds", "MSE"),
    ],
)
def test_plot_metric_per_segment_warning_non_comparable_segments(df_name, metric_name, request):
    metrics_df = request.getfixturevalue(df_name)
    with pytest.warns(UserWarning, match="Some segments have different set of folds to be aggregated on"):
        plot_metric_per_segment(metrics_df=metrics_df, metric_name=metric_name)


@pytest.mark.parametrize("plot_type", ["hist", "box", "violin"])
@pytest.mark.parametrize(
    "df_name, metric_name, per_fold_aggregation_mode",
    [
        ("metrics_df_with_folds", "MAE", None),
        ("metrics_df_with_folds", "MAE", "mean"),
        ("metrics_df_with_folds", "MAE", "median"),
        ("metrics_df_no_folds", "MAE", None),
        ("metrics_df_no_folds", "MSE", None),
    ],
)
def test_plot_metric_per_segment_ok(df_name, metric_name, per_fold_aggregation_mode, plot_type, request):
    metrics_df = request.getfixturevalue(df_name)
    metric_per_segment_distribution_plot(
        metrics_df=metrics_df,
        metric_name=metric_name,
        per_fold_aggregation_mode=per_fold_aggregation_mode,
        plot_type=plot_type,
    )


@pytest.mark.parametrize(
    "df_name, metric_name",
    [
        ("metrics_df_with_folds", "MAPE"),
        ("metrics_df_no_folds", "RMSE"),
    ],
)
def test_plot_metric_per_segment_warning_empty_segments(df_name, metric_name, request):
    metrics_df = request.getfixturevalue(df_name)
    with pytest.warns(UserWarning, match="There are segments with all missing metric values"):
        metric_per_segment_distribution_plot(metrics_df=metrics_df, metric_name=metric_name)


@pytest.mark.parametrize(
    "df_name, metric_name",
    [
        ("metrics_df_with_folds", "MSE"),
    ],
)
def test_plot_metric_per_segment_warning_non_comparable_segments(df_name, metric_name, request):
    metrics_df = request.getfixturevalue(df_name)
    with pytest.warns(UserWarning, match="Some segments have different set of folds to be aggregated on"):
        metric_per_segment_distribution_plot(metrics_df=metrics_df, metric_name=metric_name)
