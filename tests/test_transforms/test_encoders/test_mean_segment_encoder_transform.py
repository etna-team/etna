import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearMultiSegmentModel
from etna.transforms import MeanSegmentEncoderTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import select_segments_subset


def test_mean_segment_encoder_transform(mean_segment_encoder_ts, expected_mean_segment_encoder_ts):
    encoder = MeanSegmentEncoderTransform()
    transformed_df = encoder.fit_transform(mean_segment_encoder_ts).to_pandas()
    assert_frame_equal(transformed_df, expected_mean_segment_encoder_ts.to_pandas(), atol=0.01)


def test_make_future_mean_segment_encoder_transform(
    mean_segment_encoder_ts, expected_make_future_mean_segment_encoder_ts
):
    mean_segment_encoder = MeanSegmentEncoderTransform()
    mean_segment_encoder.fit_transform(mean_segment_encoder_ts)
    future_ts = mean_segment_encoder_ts.make_future(future_steps=2, transforms=[mean_segment_encoder])

    assert_frame_equal(future_ts.to_pandas(), expected_make_future_mean_segment_encoder_ts.to_pandas())


def test_not_fitted_error(mean_segment_encoder_ts):
    encoder = MeanSegmentEncoderTransform()
    with pytest.raises(ValueError, match="The transform isn't fitted"):
        encoder.transform(mean_segment_encoder_ts)


def test_new_segments_error(mean_segment_encoder_ts):
    train_ts = select_segments_subset(ts=mean_segment_encoder_ts, segments=["segment_0"])
    test_ts = select_segments_subset(ts=mean_segment_encoder_ts, segments=["segment_1"])
    transform = MeanSegmentEncoderTransform()

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_ts)


@pytest.fixture
def almost_constant_ts(random_seed) -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq=pd.offsets.Day())})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq=pd.offsets.Day())})
    df_1["segment"] = "Moscow"
    df_1["target"] = 1 + np.random.normal(0, 0.1, size=len(df_1))
    df_2["segment"] = "Omsk"
    df_2["target"] = 10 + np.random.normal(0, 0.1, size=len(df_1))
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(classic_df), freq=pd.offsets.Day())
    return ts


def test_mean_segment_encoder_forecast(almost_constant_ts):
    """Test that MeanSegmentEncoderTransform works correctly in forecast pipeline
    and helps to correctly forecast almost constant series."""
    horizon = 5
    model = LinearMultiSegmentModel()
    encoder = MeanSegmentEncoderTransform()

    train, test = almost_constant_ts.train_test_split(test_size=horizon)
    train.fit_transform([encoder])
    model.fit(train)
    future = train.make_future(horizon, transforms=[encoder])
    pred_mean_segment_encoding = model.forecast(future)
    pred_mean_segment_encoding.inverse_transform([encoder])

    metric = R2(mode="macro")

    # R2=0 => model predicts the optimal constant
    assert np.allclose(metric(pred_mean_segment_encoding, test), 0)


def test_fit_transform_with_nans(ts_diff_endings):
    encoder = MeanSegmentEncoderTransform()
    encoder.fit_transform(ts_diff_endings)


def test_save_load(almost_constant_ts):
    transform = MeanSegmentEncoderTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=almost_constant_ts)


def test_params_to_tune():
    transform = MeanSegmentEncoderTransform()
    assert len(transform.params_to_tune()) == 0
