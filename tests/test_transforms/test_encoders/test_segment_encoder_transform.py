import numpy as np
import pandas as pd
import pytest

from etna.transforms import SegmentEncoderTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import select_segments_subset


def test_segment_encoder_transform(mean_segment_encoder_ts):
    transform = SegmentEncoderTransform()
    transformed_df = transform.fit_transform(mean_segment_encoder_ts).to_pandas()
    assert (
        len(transformed_df.loc[:, pd.IndexSlice[:, "segment_code"]].columns) == 2
    ), "Number of columns not the same as segments"
    assert len(mean_segment_encoder_ts.to_pandas()) == len(transformed_df), "Row missing"
    codes = set()
    for segment in mean_segment_encoder_ts.segments:
        column = transformed_df.loc[:, pd.IndexSlice[segment, "segment_code"]]
        assert column.dtype == "category", "Column type is not category"
        assert np.all(column == column.iloc[0]), "Values are not the same for the whole column"
        codes.add(column.iloc[0])
    assert codes == {0, 1}, "Codes are not 0 and 1"


def test_not_fitted_error(mean_segment_encoder_ts):
    encoder = SegmentEncoderTransform()
    with pytest.raises(ValueError, match="The transform isn't fitted"):
        encoder.transform(mean_segment_encoder_ts)


def test_new_segments_error(mean_segment_encoder_ts):
    train_ts = select_segments_subset(ts=mean_segment_encoder_ts, segments=["segment_0"])
    test_ts = select_segments_subset(ts=mean_segment_encoder_ts, segments=["segment_1"])
    transform = SegmentEncoderTransform()

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_ts)


def test_save_load(example_tsds):
    transform = SegmentEncoderTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)


def test_params_to_tune():
    transform = SegmentEncoderTransform()
    assert len(transform.params_to_tune()) == 0
