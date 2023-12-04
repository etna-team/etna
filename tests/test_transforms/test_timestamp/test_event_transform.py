import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.metrics import MSE
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms.feature_selection import FilterFeaturesTransform
from etna.transforms.timestamp import EventTransform
from tests.test_transforms.utils import assert_sampling_is_valid


@pytest.fixture
def ts_check_event_transform_expected_binary_pre_2_post_2(ts_check_event_transform) -> TSDataset:
    periods_exog = 10
    df = ts_check_event_transform.to_pandas(features=["target"])
    df_exog = ts_check_event_transform.df_exog

    holiday = generate_const_df(start_time="2020-01-01", periods=periods_exog, freq="D", scale=1, n_segments=3)
    holiday.drop(columns=["target"], inplace=True)
    holiday["holiday_pre"] = [1, 0, 1, 0, 0, 0, 1, 1, 0, 0] + [0] * 10 + [0] * 10
    holiday["holiday_post"] = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1] + [0] * 10 + [0] * 10
    holiday = TSDataset.to_dataset(holiday).astype(float)
    df_exog = pd.concat([holiday, df_exog], axis=1)
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future="all")
    return ts


@pytest.fixture
def ts_check_event_transform_expected_binary_pre_1_post_1(ts_check_event_transform) -> TSDataset:
    periods_exog = 10
    df = ts_check_event_transform.to_pandas(features=["target"])
    df_exog = ts_check_event_transform.df_exog

    holiday = generate_const_df(start_time="2020-01-01", periods=periods_exog, freq="D", scale=1, n_segments=3)
    holiday.drop(columns=["target"], inplace=True)
    holiday["holiday_pre"] = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0] + [0] * 10 + [0] * 10
    holiday["holiday_post"] = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1] + [0] * 10 + [0] * 10
    holiday = TSDataset.to_dataset(holiday).astype(float)
    df_exog = pd.concat([holiday, df_exog], axis=1)
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future="all")
    return ts


@pytest.fixture
def ts_check_event_transform_expected_distance_pre_3_post_3(ts_check_event_transform) -> TSDataset:
    periods_exog = 10
    df = ts_check_event_transform.to_pandas(features=["target"])
    df_exog = ts_check_event_transform.df_exog

    holiday = generate_const_df(start_time="2020-01-01", periods=periods_exog, freq="D", scale=1, n_segments=3)
    holiday.drop(columns=["target"], inplace=True)
    holiday["holiday_pre"] = [1, 0, 1, 0, 0, 1 / 3, 1 / 2, 1, 0, 0] + [0] * 10 + [0] * 10
    holiday["holiday_post"] = [0, 0, 1, 0, 0, 1, 1 / 2, 1 / 3, 0, 1] + [0] * 10 + [0] * 10
    holiday = TSDataset.to_dataset(holiday).astype(float)
    df_exog = pd.concat([holiday, df_exog], axis=1)
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future="all")
    return ts


@pytest.fixture
def ts_check_event_transform() -> TSDataset:
    periods = 9
    periods_exog = periods + 1
    df = generate_const_df(start_time="2020-01-01", periods=periods, freq="D", scale=1, n_segments=3)
    df_exog = generate_const_df(start_time="2020-01-01", periods=periods_exog, freq="D", scale=1, n_segments=3)
    df_exog.rename(columns={"target": "holiday"}, inplace=True)
    df_exog["holiday"] = (
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 0] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    df = TSDataset.to_dataset(df)
    df_exog = TSDataset.to_dataset(df_exog)
    tsds = TSDataset(df, freq="D", df_exog=df_exog, known_future="all")
    return tsds


@pytest.mark.parametrize(
    "mode,n_pre,n_post,ts_expected",
    [
        ("binary", 2, 2, "ts_check_event_transform_expected_binary_pre_2_post_2"),
        ("binary", 1, 1, "ts_check_event_transform_expected_binary_pre_1_post_1"),
        ("distance", 3, 3, "ts_check_event_transform_expected_distance_pre_3_post_3"),
    ],
)
def test_fit_transform(ts_check_event_transform: TSDataset, ts_expected: TSDataset, mode, n_pre, n_post, request):
    """Check event transform correctly generate holiday features"""
    ts_expected = request.getfixturevalue(ts_expected)
    transform = EventTransform(in_column="holiday", out_column="holiday", n_pre=n_pre, n_post=n_post, mode=mode)
    result_binary = transform.fit_transform(ts=ts_check_event_transform).to_pandas()
    pd.testing.assert_frame_equal(ts_expected.to_pandas(), result_binary)


def test_input_column_not_binary(example_tsds: TSDataset):
    """Check that Exception raises when input column is not binary"""
    transform = EventTransform(in_column="target", out_column="holiday", n_pre=1, n_post=1)
    with pytest.raises(ValueError, match="Input columns must be binary"):
        transform.fit_transform(example_tsds)


@pytest.mark.parametrize("mode", ["bin", "distanced"])
def test_wrong_mode_type(mode):
    """Check that Exception raises when passed wrong mode type"""
    with pytest.raises(NotImplementedError, match=f"{mode} is not a valid ImputerMode."):
        _ = EventTransform(in_column="holiday", out_column="holiday", n_pre=1, n_post=1, mode=mode)


@pytest.mark.parametrize("n_pre,n_post", [(0, 2), (-1, -1)])
def test_wrong_distance_values(n_pre, n_post):
    """Check that Exception raises when passed wrong n_pre and n_post parameters"""
    with pytest.raises(ValueError, match=f"`n_pre` and `n_post` must be greater than zero, given {n_pre} and {n_post}"):
        _ = EventTransform(in_column="holiday", out_column="holiday", n_pre=n_pre, n_post=n_post)


def test_pipeline_with_event_transform(ts_with_binary_exog: TSDataset):
    """Check that pipeline executes without errors"""
    model = LinearPerSegmentModel()
    event = EventTransform(in_column="holiday", out_column="holiday", n_pre=1, n_post=1)
    filter_feature = FilterFeaturesTransform(exclude=["holiday"])
    pipeline = Pipeline(model=model, transforms=[event, filter_feature], horizon=10)
    pipeline.fit(ts_with_binary_exog)
    pipeline.forecast()


def test_backtest(ts_with_binary_exog: TSDataset):
    """Check that backtest function executes without errors"""
    model = LinearPerSegmentModel()
    event = EventTransform(in_column="holiday", out_column="holiday", n_pre=1, n_post=1)
    filter_feature = FilterFeaturesTransform(exclude=["holiday"])
    pipeline = Pipeline(model=model, transforms=[event, filter_feature], horizon=10)
    pipeline.backtest(ts=ts_with_binary_exog, metrics=[MSE()], n_folds=2)


@pytest.mark.parametrize("n_pre,n_post", [(1, 1), (3, 4)])
def test_params_to_tune(ts_with_binary_exog: TSDataset, n_pre: int, n_post: int):
    transform = EventTransform(in_column="holiday", out_column="holiday", n_pre=n_pre, n_post=n_post)
    assert len(transform.params_to_tune()) == 3
    assert_sampling_is_valid(transform=transform, ts=ts_with_binary_exog)
