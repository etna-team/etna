from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import max_deviation
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import rmse
from etna.metrics import sign
from etna.metrics import smape
from etna.metrics import wape
from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode
from etna.metrics.functional_metrics import count_missing_values
from etna.metrics.metrics import MAE
from etna.metrics.metrics import MAPE
from etna.metrics.metrics import MSE
from etna.metrics.metrics import MSLE
from etna.metrics.metrics import R2
from etna.metrics.metrics import RMSE
from etna.metrics.metrics import SMAPE
from etna.metrics.metrics import WAPE
from etna.metrics.metrics import MaxDeviation
from etna.metrics.metrics import MedAE
from etna.metrics.metrics import MissingCounter
from etna.metrics.metrics import Sign
from tests.utils import DummyMetric
from tests.utils import create_dummy_functional_metric


@pytest.mark.parametrize(
    "metric, expected_repr",
    (
        (MAE(), "MAE(mode = 'per-segment', missing_mode = 'error', )"),
        (MAE(mode="macro"), "MAE(mode = 'macro', missing_mode = 'error', )"),
        (MAE(missing_mode="ignore"), "MAE(mode = 'per-segment', missing_mode = 'ignore', )"),
        (MAE(mode="macro", missing_mode="ignore"), "MAE(mode = 'macro', missing_mode = 'ignore', )"),
        (MSE(), "MSE(mode = 'per-segment', missing_mode = 'error', )"),
        (RMSE(), "RMSE(mode = 'per-segment', missing_mode = 'error', )"),
        (MedAE(), "MedAE(mode = 'per-segment', missing_mode = 'error', )"),
        (MSLE(), "MSLE(mode = 'per-segment', missing_mode = 'error', )"),
        (MAPE(), "MAPE(mode = 'per-segment', missing_mode = 'error', )"),
        (SMAPE(), "SMAPE(mode = 'per-segment', missing_mode = 'error', )"),
        (R2(), "R2(mode = 'per-segment', missing_mode = 'error', )"),
        (Sign(), "Sign(mode = 'per-segment', missing_mode = 'error', )"),
        (MaxDeviation(), "MaxDeviation(mode = 'per-segment', missing_mode = 'error', )"),
        (DummyMetric(), "DummyMetric(mode = 'per-segment', alpha = 1.0, )"),
        (WAPE(), "WAPE(mode = 'per-segment', missing_mode = 'error', )"),
        (MissingCounter(), "MissingCounter(mode = 'per-segment', )"),
    ),
)
def test_repr(metric, expected_repr):
    """Check metrics __repr__ method"""
    metric_repr = metric.__repr__()
    assert metric_repr == expected_repr


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, WAPE, MissingCounter),
)
def test_name_class_name(metric_class):
    """Check metrics name property without changing its during inheritance"""
    metric_mode = "per-segment"
    metric = metric_class(mode=metric_mode)
    metric_name = metric.name
    true_name = metric_class.__name__
    assert metric_name == true_name


@pytest.mark.parametrize(
    "metric_class",
    (DummyMetric,),
)
def test_name_repr(metric_class):
    """Check metrics name property with changing its during inheritance to repr"""
    metric_mode = "per-segment"
    metric = metric_class(mode=metric_mode)
    metric_name = metric.name
    true_name = metric.__repr__()
    assert metric_name == true_name


@pytest.mark.parametrize(
    "metric_class", (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, WAPE, MissingCounter)
)
def test_metrics_macro(metric_class, train_test_dfs):
    """Check metrics interface in 'macro' mode"""
    forecast_df, true_df = train_test_dfs
    metric = metric_class(mode=MetricAggregationMode.macro)
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, float)


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_metrics_per_segment(metric_class, train_test_dfs):
    """Check metrics interface in 'per-segment' mode"""
    forecast_df, true_df = train_test_dfs
    metric = metric_class(mode=MetricAggregationMode.per_segment)
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, dict)
    for segment in forecast_df._df.columns.get_level_values("segment").unique():
        assert segment in value


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_metrics_invalid_aggregation(metric_class):
    """Check metrics behavior in case of invalid aggregation multioutput"""
    with pytest.raises(NotImplementedError):
        _ = metric_class(mode="a")


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_invalid_segments(metric_class, two_dfs_with_different_segments_sets):
    """Check metrics behavior in case of invalid segments sets"""
    forecast_df, true_df = two_dfs_with_different_segments_sets
    metric = metric_class()
    with pytest.raises(ValueError, match="There are segments in .* that are not in .*"):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_invalid_target_columns(metric_class, train_test_dfs):
    """Check metrics behavior in case of no target column in segment"""
    forecast_df, true_df = train_test_dfs
    columns = forecast_df._df.columns.to_list()
    columns[0] = ("segment_1", "not_target")
    forecast_df._df.columns = pd.MultiIndex.from_tuples(columns, names=["segment", "feature"])
    metric = metric_class()
    with pytest.raises(ValueError, match="All the segments in .* should contain 'target' column"):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_invalid_index(metric_class, two_dfs_with_different_timestamps):
    """Check metrics behavior in case of invalid index"""
    forecast_df, true_df = two_dfs_with_different_timestamps
    metric = metric_class()
    with pytest.raises(ValueError, match="y_true and y_pred have different timestamps"):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_invalid_nans_pred(metric_class, train_test_dfs):
    """Check metrics behavior in case of nans in prediction."""
    forecast_df, true_df = train_test_dfs
    forecast_df._df.iloc[0, 0] = np.NaN
    metric = metric_class()
    with pytest.raises(ValueError, match="There are NaNs in y_pred"):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize(
    "metric",
    (
        MAE(missing_mode="error"),
        MSE(missing_mode="error"),
        RMSE(missing_mode="error"),
        MedAE(missing_mode="error"),
        MSLE(missing_mode="error"),
        MAPE(missing_mode="error"),
        SMAPE(missing_mode="error"),
        R2(missing_mode="error"),
        Sign(missing_mode="error"),
        MaxDeviation(missing_mode="error"),
        DummyMetric(),
        WAPE(missing_mode="error"),
    ),
)
def test_invalid_nans_true(metric, train_test_dfs):
    """Check metrics behavior in case of nans in true values."""
    forecast_df, true_df = train_test_dfs
    true_df._df.iloc[0, 0] = np.NaN
    with pytest.raises(ValueError, match="There are NaNs in y_true"):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize(
    "metric",
    (
        MSE(missing_mode="ignore"),
        MAE(missing_mode="ignore"),
        MAPE(missing_mode="ignore"),
        SMAPE(missing_mode="ignore"),
        Sign(missing_mode="ignore"),
        WAPE(missing_mode="ignore"),
        MaxDeviation(missing_mode="ignore"),
        R2(missing_mode="ignore"),
        MedAE(missing_mode="ignore"),
        RMSE(missing_mode="ignore"),
        MSLE(missing_mode="ignore"),
        MissingCounter(),
    ),
)
def test_invalid_single_nan_ignore(metric, train_test_dfs):
    """Check metrics behavior in case of ignoring one nan in true values."""
    forecast_df, true_df = train_test_dfs
    true_df._df.iloc[0, 0] = np.NaN
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, dict)
    segments = set(forecast_df._df.columns.get_level_values("segment").unique().tolist())
    assert value.keys() == segments
    assert all(isinstance(cur_value, float) for cur_value in value.values())


@pytest.mark.parametrize(
    "metric, expected_type",
    (
        (MSE(mode="per-segment", missing_mode="ignore"), type(None)),
        (MAE(mode="per-segment", missing_mode="ignore"), type(None)),
        (MAPE(mode="per-segment", missing_mode="ignore"), type(None)),
        (SMAPE(mode="per-segment", missing_mode="ignore"), type(None)),
        (Sign(mode="per-segment", missing_mode="ignore"), type(None)),
        (WAPE(mode="per-segment", missing_mode="ignore"), type(None)),
        (MaxDeviation(mode="per-segment", missing_mode="ignore"), type(None)),
        (R2(mode="per-segment", missing_mode="ignore"), type(None)),
        (MedAE(mode="per-segment", missing_mode="ignore"), type(None)),
        (RMSE(mode="per-segment", missing_mode="ignore"), type(None)),
        (MSLE(mode="per-segment", missing_mode="ignore"), type(None)),
        (MissingCounter(mode="per-segment"), float),
    ),
)
def test_invalid_segment_nans_ignore_per_segment(metric, expected_type, train_test_dfs):
    """Check per-segment metrics behavior in case of ignoring segment of all nans in true values."""
    forecast_df, true_df = train_test_dfs
    true_df._df.iloc[:, 0] = np.NaN
    value = metric(y_true=true_df, y_pred=forecast_df)

    assert isinstance(value, dict)
    segments = set(forecast_df._df.columns.get_level_values("segment").unique().tolist())
    empty_segment = true_df._df.columns.get_level_values("segment").unique()[0]
    assert value.keys() == segments
    for cur_segment, cur_value in value.items():
        if cur_segment == empty_segment:
            assert isinstance(cur_value, expected_type)
        else:
            assert isinstance(cur_value, float)


@pytest.mark.parametrize(
    "metric",
    (
        MSE(mode="macro", missing_mode="ignore"),
        MAE(mode="macro", missing_mode="ignore"),
        MAPE(mode="macro", missing_mode="ignore"),
        SMAPE(mode="macro", missing_mode="ignore"),
        Sign(mode="macro", missing_mode="ignore"),
        WAPE(mode="macro", missing_mode="ignore"),
        MaxDeviation(mode="macro", missing_mode="ignore"),
        R2(mode="macro", missing_mode="ignore"),
        MedAE(mode="macro", missing_mode="ignore"),
        RMSE(mode="macro", missing_mode="ignore"),
        MSLE(mode="macro", missing_mode="ignore"),
        MissingCounter(mode="macro"),
    ),
)
def test_invalid_segment_nans_ignore_macro(metric, train_test_dfs):
    """Check macro metrics behavior in case of ignoring segment of all nans in true values."""
    forecast_df, true_df = train_test_dfs
    true_df._df.iloc[:, 0] = np.NaN
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, float)


@pytest.mark.parametrize(
    "metric, expected_type",
    (
        (MSE(mode="macro", missing_mode="ignore"), type(None)),
        (MAE(mode="macro", missing_mode="ignore"), type(None)),
        (MAPE(mode="macro", missing_mode="ignore"), type(None)),
        (SMAPE(mode="macro", missing_mode="ignore"), type(None)),
        (Sign(mode="macro", missing_mode="ignore"), type(None)),
        (WAPE(mode="macro", missing_mode="ignore"), type(None)),
        (MaxDeviation(mode="macro", missing_mode="ignore"), type(None)),
        (R2(mode="macro", missing_mode="ignore"), type(None)),
        (MedAE(mode="macro", missing_mode="ignore"), type(None)),
        (RMSE(mode="macro", missing_mode="ignore"), type(None)),
        (MSLE(mode="macro", missing_mode="ignore"), type(None)),
        (MissingCounter(mode="macro"), float),
    ),
)
def test_invalid_all_nans_ignore_macro(metric, expected_type, train_test_dfs):
    """Check macro metrics behavior in case of all nan values in true values."""
    forecast_df, true_df = train_test_dfs
    true_df._df.iloc[:, :] = np.NaN
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, expected_type)


@pytest.mark.parametrize(
    "metric_class, metric_fn",
    (
        (MAE, mae),
        (MSE, mse),
        (RMSE, rmse),
        (MedAE, medae),
        (MSLE, msle),
        (MAPE, mape),
        (SMAPE, smape),
        (R2, r2_score),
        (Sign, sign),
        (MaxDeviation, max_deviation),
        (DummyMetric, create_dummy_functional_metric()),
        (WAPE, wape),
        (MissingCounter, count_missing_values),
    ),
)
def test_metrics_values(metric_class, metric_fn, train_test_dfs):
    """
    Check that all the segments' metrics values in per-segments mode are equal to the same
    metric for segments' series.
    """
    forecast_df, true_df = train_test_dfs
    metric = metric_class(mode="per-segment")
    metric_values = metric(y_pred=forecast_df, y_true=true_df)
    for segment, value in metric_values.items():
        true_metric_value = metric_fn(
            y_true=true_df.loc[:, pd.IndexSlice[segment, "target"]],
            y_pred=forecast_df.loc[:, pd.IndexSlice[segment, "target"]],
        )
        assert value == true_metric_value


def _create_metric_class(metric_fn, metric_fn_signature, greater_is_better):
    def make_init(metric_fn, metric_fn_signature):
        def init(self, mode):
            Metric.__init__(self=self, mode=mode, metric_fn=metric_fn, metric_fn_signature=metric_fn_signature)

        return init

    new_class = type(
        "NewMetric",
        (Metric,),
        {
            "__init__": make_init(metric_fn=metric_fn, metric_fn_signature=metric_fn_signature),
            "greater_is_better": lambda: greater_is_better,
        },
    )

    return new_class


@pytest.mark.parametrize(
    "metric_fn, matrix_to_array_params, greater_is_better",
    (
        (mae, {"multioutput": "raw_values"}, False),
        (mse, {"multioutput": "raw_values"}, False),
        (rmse, {"multioutput": "raw_values"}, False),
        (mape, {"multioutput": "raw_values"}, False),
        (smape, {"multioutput": "raw_values"}, False),
        (medae, {"multioutput": "raw_values"}, False),
        (r2_score, {"multioutput": "raw_values"}, True),
        (sign, {"multioutput": "raw_values"}, None),
        (max_deviation, {"multioutput": "raw_values"}, False),
        (wape, {"multioutput": "raw_values"}, False),
        (count_missing_values, {"multioutput": "raw_values"}, None),
    ),
)
def test_metrics_equivalence_of_signatures(metric_fn, matrix_to_array_params, greater_is_better, train_test_dfs):
    forecast_df, true_df = train_test_dfs

    metric_1_class = _create_metric_class(
        metric_fn=metric_fn, metric_fn_signature="array_to_scalar", greater_is_better=greater_is_better
    )
    metric_1 = metric_1_class(mode="per-segment")
    metric_fn_matrix_to_array = partial(metric_fn, **matrix_to_array_params)
    metric_2_class = _create_metric_class(
        metric_fn=metric_fn_matrix_to_array, metric_fn_signature="matrix_to_array", greater_is_better=greater_is_better
    )
    metric_2 = metric_2_class(mode="per-segment")

    metric_1_values = metric_1(y_pred=forecast_df, y_true=true_df)
    metric_2_values = metric_2(y_pred=forecast_df, y_true=true_df)

    assert metric_1_values == metric_2_values


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, RMSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, MaxDeviation, DummyMetric, WAPE, MissingCounter),
)
def test_metric_values_with_changed_segment_order(metric_class, train_test_dfs):
    forecast_df, true_df = train_test_dfs
    forecast_df_new, true_df_new = deepcopy(train_test_dfs)
    segments = np.array(forecast_df.segments)

    forecast_segment_order = segments[[3, 2, 0, 1, 4]]
    forecast_df_new._df = forecast_df_new.loc[:, pd.IndexSlice[forecast_segment_order, :]]
    true_segment_order = segments[[4, 1, 3, 2, 0]]
    true_df_new._df = true_df_new.loc[:, pd.IndexSlice[true_segment_order, :]]

    metric = metric_class(mode="per-segment")
    metrics_initial = metric(y_pred=forecast_df, y_true=true_df)
    metrics_changed_order = metric(y_pred=forecast_df_new, y_true=true_df_new)

    assert metrics_initial == metrics_changed_order


@pytest.mark.parametrize(
    "metric, greater_is_better",
    (
        (MAE(), False),
        (MSE(), False),
        (RMSE(), False),
        (MedAE(), False),
        (MSLE(), False),
        (MAPE(), False),
        (SMAPE(), False),
        (R2(), True),
        (Sign(), None),
        (MaxDeviation(), False),
        (DummyMetric(), False),
        (WAPE(), False),
        (MissingCounter(), None),
    ),
)
def test_metrics_greater_is_better(metric, greater_is_better):
    assert metric.greater_is_better == greater_is_better


def test_multiple_calls():
    """Check that metric works correctly in case of multiple call."""
    timerange = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=10, freq="1D")})
    timestamp_base = pd.concat((timerange, timerange), axis=0)

    test_df_1 = timestamp_base.copy()
    test_df_2 = timestamp_base.copy()

    test_df_1["segment"] = ["A"] * 10 + ["B"] * 10
    test_df_2["segment"] = ["C"] * 10 + ["B"] * 10

    forecast_df_1 = test_df_1.copy()
    forecast_df_2 = test_df_2.copy()

    test_df_1["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    forecast_df_1["target"] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 0, 3, 4, 5, 6, 7, 8, 9]

    test_df_2["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    forecast_df_2["target"] = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    test_df_1 = test_df_1.pivot(index="timestamp", columns="segment")
    test_df_1 = test_df_1.reorder_levels([1, 0], axis=1)
    test_df_1 = test_df_1.sort_index(axis=1)
    test_df_1.columns.names = ["segment", "feature"]

    test_df_2 = test_df_2.pivot(index="timestamp", columns="segment")
    test_df_2 = test_df_2.reorder_levels([1, 0], axis=1)
    test_df_2 = test_df_2.sort_index(axis=1)
    test_df_2.columns.names = ["segment", "feature"]

    forecast_df_1 = forecast_df_1.pivot(index="timestamp", columns="segment")
    forecast_df_1 = forecast_df_1.reorder_levels([1, 0], axis=1)
    forecast_df_1 = forecast_df_1.sort_index(axis=1)
    forecast_df_1.columns.names = ["segment", "feature"]

    forecast_df_2 = forecast_df_2.pivot(index="timestamp", columns="segment")
    forecast_df_2 = forecast_df_2.reorder_levels([1, 0], axis=1)
    forecast_df_2 = forecast_df_2.sort_index(axis=1)
    forecast_df_2.columns.names = ["segment", "feature"]

    test_df_1 = TSDataset(test_df_1, freq="1D")
    test_df_2 = TSDataset(test_df_2, freq="1D")
    forecast_df_1 = TSDataset(forecast_df_1, freq="1D")
    forecast_df_2 = TSDataset(forecast_df_2, freq="1D")

    metric = MAE(mode="per-segment")
    metric_value_1 = metric(y_true=test_df_1, y_pred=forecast_df_1)

    assert sorted(metric_value_1.keys()) == ["A", "B"]
    assert metric_value_1["A"] == 0.1
    assert metric_value_1["B"] == 0.2

    metric_value_2 = metric(y_true=test_df_2, y_pred=forecast_df_2)

    assert sorted(metric_value_2.keys()) == ["B", "C"]
    assert metric_value_2["C"] == 1
    assert metric_value_2["B"] == 0
