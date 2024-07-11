import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import MSE
from etna.models import LinearMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import FilterFeaturesTransform
from etna.transforms import MeanEncoderTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import select_segments_subset


@pytest.fixture
def category_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df["target"] = [1, 2, 3, 4, np.NaN, 5] + [6, 7, 8, 9, 10, 11]

    df_exog = generate_ar_df(start_time="2001-01-01", periods=8, n_segments=2)
    df_exog.rename(columns={"target": "regressor"}, inplace=True)
    df_exog["regressor"] = ["A", "B", np.NaN, "A", pd.NA, "B", "C", "A"] + ["A", "B", "A", "A", "A", np.NaN, "A", "C"]

    ts = TSDataset(df, df_exog=df_exog, freq="D", known_future="all")
    return ts


@pytest.fixture
def expected_micro_category_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, 1, 1.5, 1.5, 2.75, 2.25] + [np.NaN, 6.0, 6.25, 7, 7.625, 8.0]

    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_micro_global_mean_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, 1, 1.5, 1.5, 2.5, 2.25] + [np.NaN, 6.0, 6.25, 7, 7.625, 8.0]

    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_micro_category_make_future_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-07", periods=2, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [3, 2.5] + [8.25, 8.5]

    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_macro_category_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, 3.5, 4, 4.875, 4, 4.85] + [np.NaN, 3.5, 3.66, 4.875, 5.5, 4.275]

    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_macro_global_mean_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, 3.5, 4, 4.875, 5, 4.85] + [np.NaN, 3.5, 3.66, 4.875, 5.5, 5.55]

    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_macro_category_make_future_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-07", periods=2, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [6, 6.33] + [6.33, 6]

    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["per-segment", "macro"])
@pytest.mark.parametrize("handle_missing", ["category", "global_mean"])
@pytest.mark.parametrize("smoothing", [1, 2])
def test_fit(category_ts, mode, handle_missing, smoothing):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode=mode,
        handle_missing=handle_missing,
        smoothing=smoothing,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit(category_ts)


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["per-segment", "macro"])
@pytest.mark.parametrize("handle_missing", ["category", "global_mean"])
@pytest.mark.parametrize("smoothing", [1, 2])
def test_fit_transform(category_ts, mode, handle_missing, smoothing):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode=mode,
        handle_missing=handle_missing,
        smoothing=smoothing,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(category_ts)


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["per-segment", "macro"])
@pytest.mark.parametrize("handle_missing", ["category", "global_mean"])
@pytest.mark.parametrize("smoothing", [1, 2])
def test_make_future(category_ts, mode, handle_missing, smoothing):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode=mode,
        handle_missing=handle_missing,
        smoothing=smoothing,
        out_column="mean_encoded_regressor",
    )
    category_ts.fit_transform([mean_encoder])
    _ = category_ts.make_future(future_steps=2, transforms=[mean_encoder])


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["per-segment", "macro"])
@pytest.mark.parametrize("handle_missing", ["category", "global_mean"])
@pytest.mark.parametrize("smoothing", [1, 2])
def test_pipeline(category_ts, mode, handle_missing, smoothing):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode=mode,
        handle_missing=handle_missing,
        smoothing=smoothing,
        out_column="mean_encoded_regressor",
    )
    filter_transform = FilterFeaturesTransform(exclude=["regressor"])
    pipeline = Pipeline(model=LinearMultiSegmentModel(), transforms=[mean_encoder, filter_transform], horizon=1)
    pipeline.backtest(category_ts, n_folds=1, metrics=[MSE()])


def test_not_fitted_error(category_ts):
    mean_encoder = MeanEncoderTransform(in_column="regressor", out_column="mean_encoded_regressor")
    with pytest.raises(ValueError, match="The transform isn't fitted"):
        mean_encoder.transform(category_ts)


def test_new_segments_error(category_ts):
    train_ts = select_segments_subset(ts=category_ts, segments=["segment_0"])
    test_ts = select_segments_subset(ts=category_ts, segments=["segment_1"])
    mean_encoder = MeanEncoderTransform(in_column="regressor", out_column="mean_encoded_regressor")

    mean_encoder.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = mean_encoder.transform(test_ts)


def test_transform_micro_category_expected(category_ts, expected_micro_category_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="per-segment",
        handle_missing="category",
        smoothing=1,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(category_ts)
    assert_frame_equal(
        category_ts.df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_micro_category_ts.df, atol=0.01
    )


def test_transform_micro_global_mean_expected(category_ts, expected_micro_global_mean_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="per-segment",
        handle_missing="global_mean",
        smoothing=1,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(category_ts)
    assert_frame_equal(
        category_ts.df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_micro_global_mean_ts.df
    )


def test_transform_micro_make_future_expected(category_ts, expected_micro_category_make_future_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="per-segment",
        handle_missing="category",
        smoothing=1,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(category_ts)
    future = category_ts.make_future(future_steps=2, transforms=[mean_encoder])

    assert_frame_equal(
        future.df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_micro_category_make_future_ts.df
    )


def test_transform_macro_category_expected(category_ts, expected_macro_category_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor", mode="macro", handle_missing="category", smoothing=1, out_column="mean_encoded_regressor"
    )
    mean_encoder.fit_transform(category_ts)
    assert_frame_equal(
        category_ts.df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_macro_category_ts.df, atol=0.01
    )


def test_transform_macro_global_mean_expected(category_ts, expected_macro_global_mean_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="macro",
        handle_missing="global_mean",
        smoothing=1,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(category_ts)
    assert_frame_equal(
        category_ts.df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_macro_global_mean_ts.df, atol=0.02
    )


def test_transform_macro_make_future_expected(category_ts, expected_macro_category_make_future_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor", mode="macro", handle_missing="category", smoothing=1, out_column="mean_encoded_regressor"
    )
    mean_encoder.fit_transform(category_ts)
    future = category_ts.make_future(future_steps=2, transforms=[mean_encoder])

    assert_frame_equal(
        future.df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]],
        expected_macro_category_make_future_ts.df,
        atol=0.01,
    )


def test_save_load(category_ts):
    mean_encoder = MeanEncoderTransform(in_column="regressor", out_column="mean_encoded_regressor")
    assert_transformation_equals_loaded_original(transform=mean_encoder, ts=category_ts)


def test_params_to_tune():
    mean_encoder = MeanEncoderTransform(in_column="regressor", out_column="mean_encoded_regressor")
    assert len(mean_encoder.params_to_tune()) == 1
