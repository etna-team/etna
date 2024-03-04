import pathlib

import numpy as np
import pytest

from etna.transforms.embeddings.models import TS2VecEmbeddingModel


@pytest.mark.smoke
def test_fit(simple_ts_with_exog):
    df = simple_ts_with_exog.to_pandas()
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    model.fit(df=df)


@pytest.mark.smoke
def test_encode_segment(simple_ts_with_exog):
    df = simple_ts_with_exog.to_pandas()
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_segment(df=df)


@pytest.mark.smoke
def test_encode_window(simple_ts_with_exog):
    df = simple_ts_with_exog.to_pandas()
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_window(df=df)


@pytest.mark.smoke
def test_save(tmp_path):
    model = TS2VecEmbeddingModel(input_dims=3)

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)


@pytest.mark.smoke
def test_load(tmp_path):
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    TS2VecEmbeddingModel.load(path=path)


def test_prepare_data(
    simple_ts_with_exog,
    expected_data=np.array(
        [
            [
                [np.NaN, np.NaN, np.NaN],
                [np.NaN, np.NaN, np.NaN],
                [30.0, 14.0, 3.0],
                [40.0, 17.0, 4.0],
                [50.0, 20.0, 5.0],
                [60.0, 23.0, 6.0],
                [70.0, 26.0, 7.0],
                [80.0, 29.0, 8.0],
                [90.0, 32.0, 9.0],
                [100.0, 35.0, 10.0],
            ],
            [
                [0.0, 5.0, 0.0],
                [10.0, 8.0, 1.0],
                [20.0, 11.0, 2.0],
                [30.0, 14.0, 3.0],
                [40.0, 17.0, 4.0],
                [50.0, 20.0, 5.0],
                [60.0, 23.0, 6.0],
                [70.0, 26.0, 7.0],
                [80.0, 29.0, 8.0],
                [90.0, 32.0, 9.0],
            ],
            [
                [90.0, 32.0, 9.0],
                [80.0, 29.0, 8.0],
                [70.0, 26.0, 7.0],
                [60.0, 23.0, 6.0],
                [50.0, 20.0, 5.0],
                [40.0, 17.0, 4.0],
                [30.0, 14.0, 3.0],
                [20.0, 11.0, 2.0],
                [10.0, 8.0, 1.0],
                [0.0, 5.0, 0.0],
            ],
            [
                [10.0, 8.0, 1.0],
                [10.0, 8.0, 1.0],
                [20.0, 11.0, 2.0],
                [20.0, 11.0, 2.0],
                [30.0, 14.0, 3.0],
                [30.0, 14.0, 3.0],
                [40.0, 17.0, 4.0],
                [40.0, 17.0, 4.0],
                [50.0, 20.0, 5.0],
                [50.0, 20.0, 5.0],
            ],
            [
                [-100.0, -25.0, -10.0],
                [-90.0, -22.0, -9.0],
                [-80.0, -19.0, -8.0],
                [-70.0, -16.0, -7.0],
                [-60.0, -13.0, -6.0],
                [-50.0, -10.0, -5.0],
                [-40.0, -7.0, -4.0],
                [-30.0, -4.0, -3.0],
                [-20.0, -1.0, -2.0],
                [-10.0, 2.0, -1.0],
            ],
        ]
    ),
):
    df = simple_ts_with_exog.to_pandas()
    model = TS2VecEmbeddingModel(input_dims=3)
    obtained_data = model._prepare_data(df=df)
    np.testing.assert_array_equal(obtained_data, expected_data)


@pytest.mark.parametrize(
    "output_dims, segment_shape_expected, window_shape_expected", [(2, (5, 2), (10, 5, 2)), (3, (5, 3), (10, 5, 3))]
)
def test_encode_format(simple_ts_with_exog, output_dims, segment_shape_expected, window_shape_expected):
    df = simple_ts_with_exog.to_pandas()
    model = TS2VecEmbeddingModel(input_dims=3, output_dims=output_dims)
    segment_embeddings = model.encode_segment(df=df)
    window_embeddings = model.encode_window(df=df)
    assert segment_embeddings.shape == segment_shape_expected
    assert window_embeddings.shape == window_shape_expected


def test_encode_pre_fitted(simple_ts_with_exog, tmp_path):
    df = simple_ts_with_exog.to_pandas()

    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    assert model._is_fitted is False

    model.fit(df=df)
    assert model._is_fitted is True

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    model_loaded = TS2VecEmbeddingModel.load(path=path)
    assert model_loaded._is_fitted is True

    np.testing.assert_array_equal(model.encode_window(df=df), model_loaded.encode_window(df=df))
    np.testing.assert_array_equal(model.encode_segment(df=df), model_loaded.encode_segment(df=df))
