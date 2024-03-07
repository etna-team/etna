import pathlib

import numpy as np
import pytest

from etna.transforms.embeddings.models import TS2VecEmbeddingModel


@pytest.mark.smoke
def test_fit(ts_with_exog_nan_begin):
    x = ts_with_exog_nan_begin.df.values
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    model.fit(x)


@pytest.mark.smoke
def test_encode_segment(ts_with_exog_nan_begin):
    x = ts_with_exog_nan_begin.df.values
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_segment(x)


@pytest.mark.smoke
def test_encode_window(ts_with_exog_nan_begin):
    x = ts_with_exog_nan_begin.df.values
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_window(x)


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
    "output_dims, segment_shape_expected, window_shape_expected", [(2, (5, 2), (10, 10)), (3, (5, 3), (10, 15))]
)
def test_encode_format(ts_with_exog_nan_begin, output_dims, segment_shape_expected, window_shape_expected):
    x = ts_with_exog_nan_begin.df.values
    model = TS2VecEmbeddingModel(input_dims=3, output_dims=output_dims)
    segment_embeddings = model.encode_segment(x)
    window_embeddings = model.encode_window(x)
    assert segment_embeddings.shape == segment_shape_expected
    assert window_embeddings.shape == window_shape_expected


def test_encode_pre_fitted(ts_with_exog_nan_begin, tmp_path):
    x = ts_with_exog_nan_begin.df.values
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    assert model._is_fitted is False

    model.fit(x)
    assert model._is_fitted is True

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    model_loaded = TS2VecEmbeddingModel.load(path=path)
    assert model_loaded._is_fitted is True

    np.testing.assert_array_equal(model.encode_window(x), model_loaded.encode_window(x))
    np.testing.assert_array_equal(model.encode_segment(x), model_loaded.encode_segment(x))


@pytest.mark.parametrize(
    "data, input_dim",
    [("ts_with_exog_nan_begin", 3), ("ts_with_exog_nan_middle", 2), ("ts_with_exog_nan_end", 1)],
)
def test_encode_not_contains_nan(data, input_dim, request):
    data = request.getfixturevalue(data).df.values
    model = TS2VecEmbeddingModel(input_dims=input_dim, n_epochs=1)
    model.fit(data)
    encoded_segment = model.encode_segment(data)
    encoded_window = model.encode_window(data)

    assert np.isnan(encoded_segment).sum() == 0
    assert np.isnan(encoded_window).sum() == 0
