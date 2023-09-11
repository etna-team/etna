from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MSE
from etna.models import NaiveModel
from etna.pipeline import Pipeline
from etna.transforms.timestamp import EventTransform
from tests.test_transforms.utils import assert_sampling_is_valid


@pytest.fixture
def ts_check_pipeline_with_event_transform(random_seed) -> TSDataset:
    periods_df = 30
    periods_df_exoc = periods_df + 10
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods_df)})
    df["segment"] = ["segment_1"] * periods_df
    df["target"] = np.arange(periods_df)

    df_exoc = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods_df_exoc)})
    df_exoc["segment"] = ["segment_1"] * periods_df_exoc
    df_exoc["holiday"] = np.random.choice([0, 1], size=periods_df_exoc)

    df = TSDataset.to_dataset(df)
    df_exoc = TSDataset.to_dataset(df_exoc)
    tsds = TSDataset(df, freq="D", df_exog=df_exoc, known_future="all")
    return tsds


@pytest.fixture
def ts_check_event_transform_expected() -> TSDataset:
    periods = 10
    dataframes = [pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)}) for _ in range(3)]
    for i in range(len(dataframes)):
        df = dataframes[i]
        df["segment"] = [f"segment_{i}"] * periods
        df["target"] = np.arange(periods)
        if i == 0:
            df["holiday"] = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            df["holiday_binary_prev"] = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
            df["holiday_binary_post"] = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1])
            df["holiday_distance_prev"] = np.array([1, 0, 1, 0, 0, 0, 0.5, 1, 0, 0])
            df["holiday_distance_post"] = np.array([0, 0, 1, 0, 0, 1, 0.5, 0, 0, 1])
        elif i == 1:
            df["holiday"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_binary_prev"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_binary_post"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_distance_prev"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_distance_post"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            df["holiday"] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            df["holiday_binary_prev"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_binary_post"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_distance_prev"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            df["holiday_distance_post"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    df = pd.concat(dataframes)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D", known_future="all")
    return tsds


@pytest.fixture
def ts_check_event_transform() -> TSDataset:
    periods = 10
    dataframes = [pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)}) for _ in range(3)]
    for i in range(len(dataframes)):
        df = dataframes[i]
        df["segment"] = [f"segment_{i}"] * periods
        df["target"] = np.arange(periods)
        if i == 0:
            df["holiday"] = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
        elif i == 1:
            df["holiday"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            df["holiday"] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    df = pd.concat(dataframes)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D", known_future="all")
    return tsds


@pytest.fixture
def ts_check_input_column_not_binary() -> TSDataset:
    periods = 5
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df["segment"] = ["segment_1"] * periods
    df["target"] = np.arange(periods)
    df["holiday"] = np.array([0, 1, 0, 0, 0.3])

    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    return tsds


def test_fit_transform(ts_check_event_transform: TSDataset, ts_check_event_transform_expected: TSDataset):
    """Check event transform works correctly generate holiday features"""
    ts_copy_distance = deepcopy(ts_check_event_transform)
    df_copy_binary_expected = ts_check_event_transform_expected.to_pandas(
        features=["segment", "target", "holiday", "holiday_binary_prev", "holiday_binary_post"]
    )
    df_copy_distance_expected = ts_check_event_transform_expected.to_pandas(
        features=["segment", "target", "holiday", "holiday_distance_prev", "holiday_distance_post"]
    )

    transform_binary = EventTransform(
        in_column="holiday", out_column="holiday_binary", n_pre=2, n_post=2, mode="binary"
    )
    transform_distance = EventTransform(
        in_column="holiday", out_column="holiday_distance", n_pre=2, n_post=2, mode="distance"
    )

    result_binary = transform_binary.fit_transform(ts=ts_check_event_transform).to_pandas()
    result_distance = transform_distance.fit_transform(ts=ts_copy_distance).to_pandas()

    pd.testing.assert_frame_equal(df_copy_binary_expected, result_binary, check_like=True, check_dtype=False)
    pd.testing.assert_frame_equal(df_copy_distance_expected, result_distance, check_like=True, check_dtype=False)


def test_input_column_not_binary(ts_check_input_column_not_binary: TSDataset):
    """Check that Exception raises when input column is not binary"""
    transform = EventTransform(in_column="holiday", out_column="holiday")
    with pytest.raises(ValueError, match="Input columns must be binary"):
        transform.fit_transform(ts_check_input_column_not_binary)


@pytest.mark.parametrize("mode", ["binaryy", "distanse"])
def test_wrong_mode_type(ts_check_pipeline_with_event_transform: TSDataset, mode):
    """Check that Exception raises when passed wrong mode type"""
    with pytest.raises(NotImplementedError, match=f"{mode} is not a valid ImputerMode."):
        _ = EventTransform(in_column="holiday", out_column="holiday", mode=mode)


@pytest.mark.parametrize("n_pre,n_post", [(0, 2), (-1, -1)])
def test_wrong_distance_values(ts_check_pipeline_with_event_transform: TSDataset, n_pre, n_post):
    """Check that Exception raises when passed wrong n_pre and n_post parameters"""
    transform = EventTransform(in_column="holiday", out_column="holiday", n_pre=n_pre, n_post=n_post)
    with pytest.raises(ValueError, match=f"`n_pre` and `n_post` must be greater than zero, given {n_pre} and {n_post}"):
        transform.fit_transform(ts_check_pipeline_with_event_transform)


def test_pipeline_with_event_transform(ts_check_pipeline_with_event_transform: TSDataset):
    """Check that pipeline executes without errors"""
    model = NaiveModel()
    transform = EventTransform(in_column="holiday", out_column="holiday")
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.fit(ts_check_pipeline_with_event_transform)
    pipeline.forecast()


def test_backtest(ts_check_pipeline_with_event_transform: TSDataset):
    """Check that backtest function executes without errors"""
    model = NaiveModel()
    transform = EventTransform(in_column="holiday", out_column="holiday")
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.backtest(ts=ts_check_pipeline_with_event_transform, metrics=[MSE()], n_folds=2)


def test_params_to_tune(ts_check_pipeline_with_event_transform: TSDataset):
    transform = EventTransform(in_column="holiday", out_column="holiday")
    assert len(transform.params_to_tune()) == 3
    assert_sampling_is_valid(transform=transform, ts=ts_check_pipeline_with_event_transform)
