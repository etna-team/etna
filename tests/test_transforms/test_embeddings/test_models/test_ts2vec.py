import pathlib

import numpy as np
import pytest

from etna.transforms.embeddings.models import TS2VecEmbeddingModel


@pytest.mark.smoke
def test_fit(simple_ts_with_exog_numpy):
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    model.fit(simple_ts_with_exog_numpy)


@pytest.mark.smoke
def test_encode_segment(simple_ts_with_exog_numpy):
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_segment(simple_ts_with_exog_numpy)


@pytest.mark.smoke
def test_encode_window(simple_ts_with_exog_numpy):
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_window(simple_ts_with_exog_numpy)


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


@pytest.mark.parametrize(
    "output_dims, segment_shape_expected, window_shape_expected", [(2, (5, 2), (10, 5, 2)), (3, (5, 3), (10, 5, 3))]
)
def test_encode_format(simple_ts_with_exog_numpy, output_dims, segment_shape_expected, window_shape_expected):
    model = TS2VecEmbeddingModel(input_dims=3, output_dims=output_dims)
    segment_embeddings = model.encode_segment(simple_ts_with_exog_numpy)
    window_embeddings = model.encode_window(simple_ts_with_exog_numpy)
    assert segment_embeddings.shape == segment_shape_expected
    assert window_embeddings.shape == window_shape_expected


def test_encode_pre_fitted(simple_ts_with_exog_numpy, tmp_path):
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    assert model._is_fitted is False

    model.fit(simple_ts_with_exog_numpy)
    assert model._is_fitted is True

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    model_loaded = TS2VecEmbeddingModel.load(path=path)
    assert model_loaded._is_fitted is True

    np.testing.assert_array_equal(
        model.encode_window(simple_ts_with_exog_numpy), model_loaded.encode_window(simple_ts_with_exog_numpy)
    )
    np.testing.assert_array_equal(
        model.encode_segment(simple_ts_with_exog_numpy), model_loaded.encode_segment(simple_ts_with_exog_numpy)
    )
