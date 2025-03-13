from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import Coverage
from etna.metrics import Width


def get_datasets_with_intervals(df, lower_name, upper_name):
    tsdf = TSDataset.to_dataset(df)
    ts_train = TSDataset(df=tsdf, freq="H")

    ts_test = TSDataset(df=tsdf.copy(), freq="H")

    intervals_df = df.rename({"target": lower_name}, axis=1)
    intervals_df[upper_name] = intervals_df[lower_name]
    intervals_df = TSDataset.to_dataset(df=intervals_df)

    intervals_df.loc[:, pd.IndexSlice["segment_1", lower_name]] = (
        intervals_df.loc[:, pd.IndexSlice["segment_1", lower_name]] + 1
    )
    intervals_df.loc[:, pd.IndexSlice["segment_1", upper_name]] = (
        intervals_df.loc[:, pd.IndexSlice["segment_1", upper_name]] + 2
    )

    ts_test.add_prediction_intervals(prediction_intervals_df=intervals_df)
    return ts_train, ts_test


@pytest.fixture
def tsdataset_with_zero_width_quantiles(example_df):
    df = TSDataset.to_dataset(example_df)
    ts_train = TSDataset(df, freq="H")

    ts_test = TSDataset(df.copy(), freq="H")

    intervals_df = pd.concat(
        [
            df.rename({"target": "target_0.025"}, axis=1, level="feature"),
            df.rename({"target": "target_0.975"}, axis=1, level="feature"),
        ],
        axis=1,
    )
    ts_test.add_prediction_intervals(prediction_intervals_df=intervals_df)

    return ts_train, ts_test


@pytest.fixture
def tsdataset_with_lower_upper_borders(example_df):
    return get_datasets_with_intervals(df=example_df, lower_name="target_lower", upper_name="target_upper")


@pytest.fixture
def tsdataset_with_quantiles_and_lower_upper_borders(example_df):
    train_ts, test_ts = get_datasets_with_intervals(df=example_df, lower_name="target_lower", upper_name="target_upper")

    intervals_df = test_ts.get_prediction_intervals()
    test_ts.drop_prediction_intervals()

    intervals_df = pd.concat(
        [
            intervals_df,
            intervals_df.rename(
                {"target_lower": "target_0.025", "target_upper": "target_0.975"}, axis=1, level="feature"
            ),
        ],
        axis=1,
    )

    test_ts.add_prediction_intervals(prediction_intervals_df=intervals_df)
    return train_ts, test_ts


@pytest.fixture
def tsdataset_with_quantiles_missing_values(tsdataset_with_zero_width_quantiles):
    _, true_ts = tsdataset_with_zero_width_quantiles
    true_ts._df.loc["2020-01-31":, pd.IndexSlice[:, "target_0.025"]] = np.NaN
    true_ts._df.loc[:"2020-01-02", pd.IndexSlice[:, "target_0.975"]] = np.NaN

    forecast_ts = deepcopy(true_ts)
    forecast_ts._df.fillna(0, inplace=True)

    return forecast_ts, true_ts


@pytest.fixture
def tsdataset_with_borders_missing_values(tsdataset_with_lower_upper_borders):
    _, true_ts = tsdataset_with_lower_upper_borders
    true_ts._df.loc["2020-01-31":, pd.IndexSlice[:, "target_lower"]] = np.NaN
    true_ts._df.loc[:"2020-01-02", pd.IndexSlice[:, "target_upper"]] = np.NaN

    forecast_ts = deepcopy(true_ts)
    forecast_ts._df.fillna(0, inplace=True)

    return forecast_ts, true_ts


@pytest.fixture
def tsdataset_with_intervals_and_missing_values(tsdataset_with_quantiles_and_lower_upper_borders):
    _, true_ts = tsdataset_with_quantiles_and_lower_upper_borders
    true_ts._df.loc["2020-01-31":, pd.IndexSlice[:, "target"]] = np.NaN
    true_ts._df.loc[:"2020-01-02", pd.IndexSlice[:, "target"]] = np.NaN

    forecast_ts = deepcopy(true_ts)
    forecast_ts._df.fillna(0, inplace=True)

    return forecast_ts, true_ts


@pytest.fixture
def tsdataset_with_intervals_and_missing_segment(tsdataset_with_quantiles_and_lower_upper_borders):
    _, true_ts = tsdataset_with_quantiles_and_lower_upper_borders
    true_ts._df.loc[:, pd.IndexSlice["segment_1", "target"]] = np.NaN

    forecast_ts = deepcopy(true_ts)
    forecast_ts._df.fillna(0, inplace=True)

    return forecast_ts, true_ts


@pytest.mark.parametrize(
    "metric, expected_repr",
    (
        (
            Coverage(),
            "Coverage(quantiles = [0.025, 0.975], mode = 'per-segment', upper_name = None, lower_name = None, missing_mode = 'error', )",
        ),
        (
            Coverage(mode="macro"),
            "Coverage(quantiles = [0.025, 0.975], mode = 'macro', upper_name = None, lower_name = None, missing_mode = 'error', )",
        ),
        (
            Coverage(mode="macro"),
            "Coverage(quantiles = [0.025, 0.975], mode = 'macro', upper_name = None, lower_name = None, missing_mode = 'error', )",
        ),
        (
            Coverage(missing_mode="ignore"),
            "Coverage(quantiles = [0.025, 0.975], mode = 'per-segment', upper_name = None, lower_name = None, missing_mode = 'ignore', )",
        ),
        (
            Coverage(mode="macro", missing_mode="ignore"),
            "Coverage(quantiles = [0.025, 0.975], mode = 'macro', upper_name = None, lower_name = None, missing_mode = 'ignore', )",
        ),
        (
            Width(),
            "Width(quantiles = [0.025, 0.975], mode = 'per-segment', upper_name = None, lower_name = None, missing_mode = 'error', )",
        ),
    ),
)
def test_repr(metric, expected_repr):
    """Check metrics __repr__ method"""
    metric_repr = metric.__repr__()
    assert metric_repr == expected_repr


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("upper_name,lower_name", ((None, "name"), ("name", None)))
def test_single_border_name_set_error(metric_class, upper_name, lower_name):
    with pytest.raises(ValueError, match="Both `lower_name` and `upper_name` must be set"):
        _ = metric_class(lower_name=lower_name, upper_name=upper_name)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("quantiles,upper_name,lower_name", (((0.025, 0.975), "target_upper", "target_lower"),))
def test_interval_names_and_quantiles_set_error(
    tsdataset_with_lower_upper_borders, metric_class, quantiles, upper_name, lower_name
):
    with pytest.raises(ValueError, match="Both `quantiles` and border names are specified"):
        _ = metric_class(quantiles=quantiles, lower_name=lower_name, upper_name=upper_name)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize(
    "upper_name,lower_name", (("name1", "name2"), ("target_upper", "name"), ("name", "target_lower"))
)
def test_interval_names_not_in_dataset_error(tsdataset_with_lower_upper_borders, metric_class, upper_name, lower_name):
    train_ts, test_ts = tsdataset_with_lower_upper_borders
    metric = metric_class(quantiles=None, lower_name=lower_name, upper_name=upper_name)
    with pytest.raises(ValueError, match="Provided intervals borders names must be in dataset!"):
        _ = metric(y_true=train_ts, y_pred=test_ts)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("quantiles", ((0.025,), tuple(), (0.1, 0.2, 0.3)))
def test_quantiles_invalid_size_error(tsdataset_with_lower_upper_borders, metric_class, quantiles):
    with pytest.raises(ValueError, match="Expected tuple with two values"):
        _ = metric_class(quantiles=quantiles, lower_name=None, upper_name=None)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("quantiles", ((0.025, 0.975), (0.1, 0.5)))
def test_quantiles_not_presented_error(tsdataset_with_lower_upper_borders, metric_class, quantiles):
    train_ts, test_ts = tsdataset_with_lower_upper_borders
    metric = metric_class(quantiles=quantiles, lower_name=None, upper_name=None)
    with pytest.raises(ValueError, match="All quantiles must be presented in the dataset!"):
        _ = metric(y_true=train_ts, y_pred=test_ts)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
def test_no_intervals_error(example_tsds, metric_class):
    with pytest.raises(ValueError, match="All quantiles must be presented in the dataset!"):
        _ = metric_class()(y_true=example_tsds, y_pred=example_tsds)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
def test_missing_values_in_quantiles_error(tsdataset_with_quantiles_missing_values, metric_class):
    _, ts = tsdataset_with_quantiles_missing_values
    with pytest.raises(ValueError, match="Quantiles contain missing values!"):
        _ = metric_class()(y_true=ts, y_pred=ts)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
def test_missing_values_in_borders_error(tsdataset_with_borders_missing_values, metric_class):
    _, ts = tsdataset_with_borders_missing_values
    with pytest.raises(ValueError, match="Provided intervals borders contain missing values!"):
        _ = metric_class(lower_name="target_lower", upper_name="target_upper")(y_true=ts, y_pred=ts)


def test_width_metric_with_zero_width_quantiles(tsdataset_with_zero_width_quantiles):
    ts_train, ts_test = tsdataset_with_zero_width_quantiles

    expected_metric = 0.0
    width_metric = Width(mode="per-segment")(ts_train, ts_test)

    for segment in width_metric:
        assert width_metric[segment] == expected_metric


@pytest.mark.parametrize(
    "quantiles,lower_name,upper_name",
    (
        (None, "target_0.025", "target_0.975"),
        (None, "target_lower", "target_upper"),
        ((0.025, 0.975), None, None),
        (None, None, None),
    ),
)
def test_width_metric_with_different_width_and_shifted_quantiles(
    tsdataset_with_quantiles_and_lower_upper_borders, quantiles, lower_name, upper_name
):
    ts_train, ts_test = tsdataset_with_quantiles_and_lower_upper_borders

    expected_metric = {"segment_1": 1.0, "segment_2": 0.0}
    width_metric = Width(mode="per-segment", quantiles=quantiles, lower_name=lower_name, upper_name=upper_name)(
        ts_train, ts_test
    )

    for segment in width_metric:
        assert width_metric[segment] == expected_metric[segment]


@pytest.mark.parametrize(
    "quantiles,lower_name,upper_name",
    (
        (None, "target_0.025", "target_0.975"),
        (None, "target_lower", "target_upper"),
        ((0.025, 0.975), None, None),
        (None, None, None),
    ),
)
def test_coverage_metric_with_different_width_and_shifted_quantiles(
    tsdataset_with_quantiles_and_lower_upper_borders, quantiles, lower_name, upper_name
):
    ts_train, ts_test = tsdataset_with_quantiles_and_lower_upper_borders

    expected_metric = {"segment_1": 0.0, "segment_2": 1.0}
    coverage_metric = Coverage(mode="per-segment", quantiles=quantiles, lower_name=lower_name, upper_name=upper_name)(
        ts_train, ts_test
    )

    for segment in coverage_metric:
        assert coverage_metric[segment] == expected_metric[segment]


@pytest.mark.parametrize("metric", [Coverage(quantiles=(0.1, 0.3)), Width(quantiles=(0.1, 0.3))])
def test_using_not_presented_quantiles(metric, tsdataset_with_zero_width_quantiles):
    ts_train, ts_test = tsdataset_with_zero_width_quantiles
    with pytest.raises(ValueError, match="All quantiles must be presented in the dataset!"):
        _ = metric(ts_train, ts_test)


@pytest.mark.parametrize(
    "metric, greater_is_better", ((Coverage(quantiles=(0.1, 0.3)), None), (Width(quantiles=(0.1, 0.3)), False))
)
def test_metrics_greater_is_better(metric, greater_is_better):
    assert metric.greater_is_better == greater_is_better


@pytest.mark.parametrize(
    "metric_class",
    (Coverage, Width),
)
def test_missing_values_in_pred_error(metric_class, tsdataset_with_intervals_and_missing_values):
    true_ts, forecast_ts = tsdataset_with_intervals_and_missing_values
    metric = metric_class()
    with pytest.raises(ValueError, match="There are NaNs in y_pred"):
        _ = metric(y_true=true_ts, y_pred=forecast_ts)


@pytest.mark.parametrize(
    "metric",
    (
        Coverage(missing_mode="error"),
        Width(missing_mode="error"),
    ),
)
def test_missing_values_in_true_error(metric, tsdataset_with_intervals_and_missing_values):
    forecast_ts, true_ts = tsdataset_with_intervals_and_missing_values
    with pytest.raises(ValueError, match="There are NaNs in y_true"):
        _ = metric(y_true=true_ts, y_pred=forecast_ts)


@pytest.mark.parametrize(
    "metric, expected_type",
    (
        (Coverage(mode="per-segment", missing_mode="ignore"), type(None)),
        (Width(mode="per-segment", missing_mode="ignore"), float),
    ),
)
@pytest.mark.parametrize(
    "dataset_name, empty_segment",
    (
        ("tsdataset_with_intervals_and_missing_values", None),
        ("tsdataset_with_intervals_and_missing_segment", "segment_1"),
    ),
)
def test_missing_values_ignore_per_segment(metric, dataset_name, empty_segment, expected_type, request):
    forecast_ts, true_ts = request.getfixturevalue(dataset_name)
    segments = set(forecast_ts.segments)

    value = metric(y_true=true_ts, y_pred=forecast_ts)

    assert isinstance(value, dict)
    assert value.keys() == segments

    if empty_segment is not None:
        assert isinstance(value[empty_segment], expected_type)
        value.pop(empty_segment)

    assert all(isinstance(cur_value, float) for cur_value in value.values())


@pytest.mark.parametrize(
    "metric",
    (
        Coverage(mode="macro", missing_mode="ignore"),
        Width(mode="macro", missing_mode="ignore"),
    ),
)
@pytest.mark.parametrize(
    "dataset_name", ("tsdataset_with_intervals_and_missing_values", "tsdataset_with_intervals_and_missing_segment")
)
def test_segment_all_missing_ignore_macro(metric, dataset_name, request):
    forecast_df, true_df = request.getfixturevalue(dataset_name)
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, float)


@pytest.mark.parametrize(
    "metric,expected_type",
    ((Coverage(mode="macro", missing_mode="ignore"), type(None)), (Width(mode="macro", missing_mode="ignore"), float)),
)
def test_all_missing_values_ignore_macro(metric, tsdataset_with_intervals_and_missing_segment, expected_type):
    forecast_ts, true_ts = tsdataset_with_intervals_and_missing_segment
    true_ts._df.iloc[:, :] = np.NaN
    value = metric(y_true=true_ts, y_pred=forecast_ts)
    assert isinstance(value, expected_type)


@pytest.mark.parametrize(
    "dataset_name, upper_name, lower_name",
    (
        ("tsdataset_with_intervals_and_missing_values", None, None),
        ("tsdataset_with_borders_missing_values", "target_lower", "target_upper"),
    ),
)
@patch("etna.metrics.MetricWithMissingHandling._validate_nans")
@patch("etna.metrics.intervals_metrics._IntervalsMetricMixin._validate_tsdataset_intervals")
def test_mocked_width_missing_values_handling(
    pred_check_mock, intervals_check_mock, dataset_name, upper_name, lower_name, request
):
    true_ts, forecast_ts = request.getfixturevalue(dataset_name)

    metric = Width(missing_mode="ignore", lower_name=lower_name, upper_name=upper_name)
    result = metric(y_true=true_ts, y_pred=forecast_ts)
    assert result == {"segment_1": 1.0, "segment_2": 0.0}
