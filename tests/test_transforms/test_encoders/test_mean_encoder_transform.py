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

    ts = TSDataset(df, df_exog=df_exog, freq=pd.offsets.Day(), known_future="all")
    return ts


@pytest.fixture
def expected_micro_category_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, np.NaN, np.NaN, 1.5, 2.75, 2.25] + [np.NaN, np.NaN, 6.25, 7, 7.625, np.NaN]
    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_micro_global_mean_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, np.NaN, 1.5, 1.5, 2.5, 2.25] + [np.NaN, np.NaN, 6.25, 7, 7.625, 8.0]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_micro_category_make_future_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-07", periods=2, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [3, 2.5] + [8.25, 8.5]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_macro_category_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, np.NaN, np.NaN, 4.875, 4, 4.851] + [np.NaN, np.NaN, 3.66, 4.875, 5.5, 4.27]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_macro_global_mean_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, np.NaN, 4, 4.875, 5, 4.85] + [np.NaN, np.NaN, 3.66, 4.875, 5.5, 5.55]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_macro_category_make_future_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-07", periods=2, n_segments=2)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [6, 6.33] + [6.33, 6]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_begin_nan() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=1)
    df["target"] = [np.NaN, 1, 2, 3, 4, 5]

    df_exog = generate_ar_df(start_time="2001-01-01", periods=8, n_segments=1)
    df_exog.rename(columns={"target": "regressor"}, inplace=True)
    df_exog["regressor"] = ["A", "B", "A", "A", "B", "B", "C", "A"]

    ts = TSDataset(df, df_exog=df_exog, freq=pd.offsets.Day(), known_future="all")
    return ts


@pytest.fixture
def expected_ts_begin_nan_smooth_1() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=1)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, np.NaN, np.NaN, 1.75, 1.5, 2.5]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_ts_begin_nan_smooth_2() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=1)
    df.rename(columns={"target": "mean_encoded_regressor"}, inplace=True)
    df["mean_encoded_regressor"] = [np.NaN, np.NaN, np.NaN, 5 / 3, 5 / 3, 2.5]

    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def multiple_nan_target_category_ts() -> TSDataset:
    """Fixture with segment having multiple NaN targets:

    * For `regressor="A"` set of NaN timestamp goes before first notna value
    * For `regressor="B"` set of NaN timestamp goes after first notna value
    """
    df = generate_ar_df(n_segments=1, start_time="2001-01-01", periods=8)
    df["target"] = [np.nan, 1.5, np.nan, 3.0, 4.0, np.NaN, np.NaN, np.NaN]

    df_exog = generate_ar_df(n_segments=1, start_time="2001-01-01", periods=9)
    df_exog.rename(columns={"target": "regressor"}, inplace=True)
    df_exog["regressor"] = ["A", "B", "A", "A", "B", "B", "B", "A", "A"]

    ts = TSDataset(df=df, df_exog=df_exog, freq=pd.offsets.Day(), known_future="all")

    return ts


@pytest.fixture
def expected_multiple_nan_target_category_ts() -> TSDataset:
    df = generate_ar_df(n_segments=1, start_time="2001-01-01", periods=8)
    df.rename(columns={"target": "regressor_mean"}, inplace=True)
    df["regressor_mean"] = [np.NaN, np.NaN, np.NaN, np.NaN, 1.5, 2.75, 2.75, 3.0]

    ts = TSDataset(df=df, freq=pd.offsets.Day())

    return ts


@pytest.fixture
def mean_segment_encoder_ts(mean_segment_encoder_ts) -> TSDataset:
    df = generate_ar_df(n_segments=2, start_time="2001-01-01", periods=7)
    df = df.drop(columns=["target"])
    df["segment_feature"] = ["segment_0"] * 7 + ["segment_1"] * 7
    df_wide = TSDataset.to_dataset(df)
    mean_segment_encoder_ts.add_features_from_pandas(df_wide, update_exog=True, regressors=["segment_feature"])

    return mean_segment_encoder_ts


@pytest.fixture
def multiple_nan_target_two_segments_ts() -> TSDataset:
    """Fixture with two segments having multiple NaN targets:

    * For `regressor="A"` set of NaN timestamp goes before first notna value
    * For `regressor="B"` set of NaN timestamp goes after first notna value
    """
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df["target"] = [np.NaN, 2, np.NaN, 4, np.NaN, 5] + [np.NaN, 7, np.NaN, np.NaN, 10, 11]

    df_exog = generate_ar_df(start_time="2001-01-01", periods=7, n_segments=2)
    df_exog.rename(columns={"target": "regressor"}, inplace=True)
    df_exog["regressor"] = ["A", "B", "A", "A", "B", "B", "A"] + ["A", "B", "A", "B", "A", "B", "A"]

    ts = TSDataset(df, df_exog=df_exog, freq=pd.offsets.Day(), known_future="all")

    return ts


@pytest.fixture
def expected_multiple_nan_target_two_segments_ts() -> TSDataset:
    df = generate_ar_df(start_time="2001-01-01", periods=6, n_segments=2)
    df.rename(columns={"target": "regressor_mean"}, inplace=True)
    df["regressor_mean"] = [np.NaN, np.NaN, np.NaN, np.NaN, 4.5, 4.5] + [np.NaN, np.NaN, np.NaN, 4.5, 4, 4.5]

    ts = TSDataset(df=df, freq=pd.offsets.Day())

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
        category_ts._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_micro_category_ts._df, atol=0.01
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
        category_ts._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_micro_global_mean_ts._df
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
        future._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_micro_category_make_future_ts._df
    )


def test_transform_macro_category_expected(category_ts, expected_macro_category_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor", mode="macro", handle_missing="category", smoothing=1, out_column="mean_encoded_regressor"
    )
    mean_encoder.fit_transform(category_ts)
    assert_frame_equal(
        category_ts._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_macro_category_ts._df, atol=0.01
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
        category_ts._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]], expected_macro_global_mean_ts._df, atol=0.02
    )


def test_transform_macro_make_future_expected(category_ts, expected_macro_category_make_future_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor", mode="macro", handle_missing="category", smoothing=1, out_column="mean_encoded_regressor"
    )
    mean_encoder.fit_transform(category_ts)
    future = category_ts.make_future(future_steps=2, transforms=[mean_encoder])

    assert_frame_equal(
        future._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]],
        expected_macro_category_make_future_ts._df,
        atol=0.01,
    )


def test_ts_begin_nan_smooth_1(ts_begin_nan, expected_ts_begin_nan_smooth_1):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="per-segment",
        handle_missing="category",
        smoothing=1,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(ts_begin_nan)
    assert_frame_equal(
        ts_begin_nan._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]],
        expected_ts_begin_nan_smooth_1._df,
        atol=0.01,
    )


def test_ts_begin_nan_smooth_2(ts_begin_nan, expected_ts_begin_nan_smooth_2):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="per-segment",
        handle_missing="category",
        smoothing=2,
        out_column="mean_encoded_regressor",
    )
    mean_encoder.fit_transform(ts_begin_nan)
    assert_frame_equal(
        ts_begin_nan._df.loc[:, pd.IndexSlice[:, "mean_encoded_regressor"]],
        expected_ts_begin_nan_smooth_2._df,
        atol=0.01,
    )


def test_mean_segment_encoder(mean_segment_encoder_ts, expected_mean_segment_encoder_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="segment_feature",
        mode="per-segment",
        handle_missing="category",
        smoothing=0,
        out_column="segment_mean",
    )
    mean_encoder.fit_transform(mean_segment_encoder_ts)
    assert_frame_equal(
        mean_segment_encoder_ts._df.loc[:, pd.IndexSlice[:, "segment_mean"]],
        expected_mean_segment_encoder_ts._df.loc[:, pd.IndexSlice[:, "segment_mean"]],
        atol=0.01,
    )


def test_multiple_nan_target_category_ts(multiple_nan_target_category_ts, expected_multiple_nan_target_category_ts):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="per-segment",
        handle_missing="category",
        smoothing=0,
        out_column="regressor_mean",
    )
    mean_encoder.fit_transform(multiple_nan_target_category_ts)
    assert_frame_equal(
        multiple_nan_target_category_ts._df.loc[:, pd.IndexSlice[:, "regressor_mean"]],
        expected_multiple_nan_target_category_ts._df,
        atol=0.01,
    )


def test_multiple_nan_target_two_segments_ts(
    multiple_nan_target_two_segments_ts, expected_multiple_nan_target_two_segments_ts
):
    mean_encoder = MeanEncoderTransform(
        in_column="regressor",
        mode="macro",
        handle_missing="category",
        smoothing=0,
        out_column="regressor_mean",
    )
    mean_encoder.fit_transform(multiple_nan_target_two_segments_ts)
    assert_frame_equal(
        multiple_nan_target_two_segments_ts._df.loc[:, pd.IndexSlice[:, "regressor_mean"]],
        expected_multiple_nan_target_two_segments_ts._df,
        atol=0.01,
    )


def test_save_load(category_ts):
    mean_encoder = MeanEncoderTransform(in_column="regressor", out_column="mean_encoded_regressor")
    assert_transformation_equals_loaded_original(transform=mean_encoder, ts=category_ts)


def test_params_to_tune():
    mean_encoder = MeanEncoderTransform(in_column="regressor", out_column="mean_encoded_regressor")
    assert len(mean_encoder.params_to_tune()) == 1
