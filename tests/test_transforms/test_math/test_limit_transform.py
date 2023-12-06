import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MSE
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms.math import LimitTransform


def transform_left_bound(features: pd.Series, lower_bound=None, upper_bound=None) -> pd.Series:
    return np.log(features - lower_bound + 1e-10)


def transform_right_bound(features: pd.Series, lower_bound=None, upper_bound=None) -> pd.Series:
    return np.log(upper_bound + 1e-10 - features)


def transform_both_bounds(features: pd.Series, lower_bound=None, upper_bound=None) -> pd.Series:
    return np.log((features - lower_bound + 1e-10) / (upper_bound + 1e-10 - features))


def transform_no_bounds(features: pd.Series, lower_bound=None, upper_bound=None) -> pd.Series:
    return features


@pytest.fixture
def ts_check_pipeline_with_limit_transform() -> TSDataset:
    periods = 100
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df["segment"] = ["segment_1"] * periods
    df["target"] = np.arange(periods)

    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    return tsds


@pytest.fixture
def ts_check_limit_transform(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = ["segment_1"] * periods
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1["non_target"] = np.random.uniform(10, 20, size=periods)
    df1.loc[5, "target"] = np.NaN

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = ["segment_2"] * periods
    df2["target"] = np.random.uniform(-10, 0, size=periods)
    df2["non_target"] = np.random.uniform(-10, 0, size=periods)
    df2.loc[10, "target"] = np.NaN

    df = pd.concat((df1, df2))
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    return tsds


@pytest.mark.parametrize(
    "lower_bound,upper_bound,in_column,transform",
    [
        (None, None, "target", transform_no_bounds),
        (None, 20, "target", transform_right_bound),
        (-10, None, "target", transform_left_bound),
        (-10, 20, "target", transform_both_bounds),
        (None, None, "non_target", transform_no_bounds),
        (None, 20, "non_target", transform_right_bound),
        (-10, None, "non_target", transform_left_bound),
        (-10, 20, "non_target", transform_both_bounds),
    ],
)
def test_fit_transform(
    ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, in_column: str, transform
):
    """Check forward scaled logit transform works correctly on target column"""
    df_copy = ts_check_limit_transform.to_pandas()
    df_copy.loc[:, pd.IndexSlice[["segment_1"], in_column]] = df_copy["segment_1"][in_column].apply(
        transform, lower_bound=lower_bound, upper_bound=upper_bound
    )
    df_copy.loc[:, pd.IndexSlice[["segment_2"], in_column]] = df_copy["segment_2"][in_column].apply(
        transform, lower_bound=lower_bound, upper_bound=upper_bound
    )
    preprocess = LimitTransform(in_column=in_column, lower_bound=lower_bound, upper_bound=upper_bound)
    result = preprocess.fit_transform(ts=ts_check_limit_transform).to_pandas()

    pd.testing.assert_frame_equal(df_copy, result)


@pytest.mark.parametrize(
    "lower_bound,upper_bound,in_column",
    [
        (None, None, "target"),
        (None, 20, "target"),
        (-10, None, "target"),
        (-10, 20, "target"),
        (None, None, "non_target"),
        (None, 20, "non_target"),
        (-10, None, "non_target"),
        (-10, 20, "non_target"),
    ],
)
def test_inverse_transform(ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, in_column: str):
    """Check that inverse_transform rolls back transform result"""
    df_copy = ts_check_limit_transform.to_pandas()
    preprocess = LimitTransform(in_column=in_column, lower_bound=lower_bound, upper_bound=upper_bound)
    preprocess.fit_transform(ts=ts_check_limit_transform)
    preprocess.inverse_transform(ts=ts_check_limit_transform)
    result = ts_check_limit_transform.to_pandas()
    pd.testing.assert_frame_equal(df_copy, result)


# TODO: https://github.com/etna-team/etna/issues/66
@pytest.mark.xfail
def test_inverse_transform_fail(ts_check_limit_transform: TSDataset):
    """Check that inverse_transform rolls back transform result"""
    df_copy = ts_check_limit_transform.to_pandas()
    preprocess = LimitTransform(in_column="target", lower_bound=None, upper_bound=20)
    preprocess.fit_transform(ts=ts_check_limit_transform)
    preprocess.inverse_transform(ts=ts_check_limit_transform)
    result = ts_check_limit_transform.to_pandas()
    pd.testing.assert_frame_equal(df_copy, result, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize(
    "lower_bound,upper_bound,left_border,right_border",
    [(-10, 0, -10, 0), (10, 20, 10, 20), (0, 10, 0, 10), (None, 0, np.NINF, 0), (0, None, 0, np.inf)],
)
def test_fit_transform_values_out_of_borders(
    ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, left_border: float, right_border: float
):
    """Check that Exception raises when there are values out of bounds"""
    transform = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    with pytest.raises(ValueError, match=f"Detected values out \[{left_border}, {right_border}\]"):
        transform.fit_transform(ts_check_limit_transform)


def test_pipeline_with_limit_transform(ts_check_pipeline_with_limit_transform: TSDataset):
    """
    Check that forecasted target and quantiles are in given range with LimitTransform
    """
    model = ProphetModel()
    transform = LimitTransform(in_column="target", lower_bound=0, upper_bound=105)
    pipeline_with_limit = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline_without_limit = Pipeline(model=model, transforms=[], horizon=10)

    pipeline_with_limit.fit(ts_check_pipeline_with_limit_transform)
    pipeline_without_limit.fit(ts_check_pipeline_with_limit_transform)

    forecast_ts_limited = pipeline_with_limit.forecast(prediction_interval=True)
    forecast_ts_unlimited = pipeline_without_limit.forecast(prediction_interval=True)

    forecast_df_limited = forecast_ts_limited.to_pandas()
    forecast_df_unlimited = forecast_ts_unlimited.to_pandas()
    # TODO: https://github.com/etna-team/etna/issues/66
    assert (forecast_df_limited >= 0).all().all() and (forecast_df_limited <= 105.00001).all().all()
    assert (forecast_df_unlimited < 0).any().any() or (forecast_df_unlimited > 105).any().any()


def test_backtest(ts_check_pipeline_with_limit_transform: TSDataset):
    """Check that backtest function executes without errors"""
    model = NaiveModel()
    transform = LimitTransform(in_column="target", lower_bound=0, upper_bound=105)
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.backtest(ts=ts_check_pipeline_with_limit_transform, metrics=[MSE()], n_folds=2)
