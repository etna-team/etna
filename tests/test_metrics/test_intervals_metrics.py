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


@pytest.fixture()
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


@pytest.fixture()
def tsdataset_with_lower_upper_borders(example_df):
    return get_datasets_with_intervals(df=example_df, lower_name="target_lower", upper_name="target_upper")


@pytest.fixture()
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


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("upper_name,lower_name", ((None, "name"), ("name", None)))
def test_single_border_name_set_error(metric_class, upper_name, lower_name):
    with pytest.raises(ValueError, match="Both `lower_name` and `upper_name` must be set"):
        _ = metric_class(lower_name=lower_name, upper_name=upper_name)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("quantiles", ((0.025, 0.975), tuple()))
@pytest.mark.parametrize(
    "upper_name,lower_name", (("name1", "name2"), ("target_upper", "name"), ("name", "target_lower"))
)
def test_interval_names_not_in_dataset_error(
    tsdataset_with_lower_upper_borders, metric_class, quantiles, upper_name, lower_name
):
    train_ts, test_ts = tsdataset_with_lower_upper_borders
    metric = metric_class(quantiles=quantiles, lower_name=lower_name, upper_name=upper_name)
    with pytest.raises(ValueError, match="Provided intervals borders names must be in dataset!"):
        _ = metric(y_true=train_ts, y_pred=test_ts)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("quantiles", ((0.025, 0.975), tuple()))
def test_quantiles_not_presented_names_not_set_error(tsdataset_with_lower_upper_borders, metric_class, quantiles):
    train_ts, test_ts = tsdataset_with_lower_upper_borders
    metric = metric_class(quantiles=quantiles, lower_name=None, upper_name=None)
    with pytest.raises(ValueError, match="All quantiles must be presented in the dataset!"):
        _ = metric(y_true=train_ts, y_pred=test_ts)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
@pytest.mark.parametrize("quantiles,upper_name, lower_name", (((0.025, 0.975), "target_upper", "target_lower"),))
def test_two_intervals_specified_error(
    tsdataset_with_quantiles_and_lower_upper_borders, metric_class, quantiles, upper_name, lower_name
):
    train_ts, test_ts = tsdataset_with_quantiles_and_lower_upper_borders
    metric = metric_class(quantiles=quantiles, lower_name=lower_name, upper_name=upper_name)
    with pytest.raises(ValueError, match="Quantiles and border names are both set and point to different intervals!"):
        _ = metric(y_true=train_ts, y_pred=test_ts)


@pytest.mark.parametrize("metric_class", (Coverage, Width))
def test_no_intervals_error(example_tsds, metric_class):
    with pytest.raises(ValueError, match="All quantiles must be presented in the dataset!"):
        _ = metric_class()(y_true=example_tsds, y_pred=example_tsds)


def test_width_metric_with_zero_width_quantiles(tsdataset_with_zero_width_quantiles):
    ts_train, ts_test = tsdataset_with_zero_width_quantiles

    expected_metric = 0.0
    width_metric = Width(mode="per-segment")(ts_train, ts_test)

    for segment in width_metric:
        assert width_metric[segment] == expected_metric


@pytest.mark.parametrize(
    "quantiles,lower_name,upper_name",
    (
        ((0.025, 0.975), "target_0.025", "target_0.975"),
        ((), "target_0.025", "target_0.975"),
        ((), "target_lower", "target_upper"),
        ((0.1, 0.9), "target_lower", "target_upper"),
        ((0.025, 0.975), None, None),
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
        ((0.025, 0.975), "target_0.025", "target_0.975"),
        ((), "target_0.025", "target_0.975"),
        ((), "target_lower", "target_upper"),
        ((0.1, 0.9), "target_lower", "target_upper"),
        ((0.025, 0.975), None, None),
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
