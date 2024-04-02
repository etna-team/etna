import pathlib

import numpy as np
import pytest

from etna.transforms.embeddings.models import TSTCCEmbeddingModel


@pytest.mark.smoke
def test_fit(ts_with_exog_nan_begin_numpy):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=3)
    model.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)


@pytest.mark.smoke
def test_encode_segment(ts_with_exog_nan_begin_numpy):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=3)
    model.encode_segment(ts_with_exog_nan_begin_numpy)


@pytest.mark.smoke
def test_encode_window(ts_with_exog_nan_begin_numpy):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=3)
    model.encode_window(ts_with_exog_nan_begin_numpy)


@pytest.mark.smoke
def test_save(tmp_path):
    model = TSTCCEmbeddingModel(input_dims=3)

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)


@pytest.mark.smoke
def test_load(tmp_path):
    model = TSTCCEmbeddingModel(input_dims=3)

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    TSTCCEmbeddingModel.load(path=path)


@pytest.mark.parametrize(
    "output_dims, segment_shape_expected, window_shape_expected", [(2, (5, 2), (5, 10, 2)), (3, (5, 3), (5, 10, 3))]
)
def test_encode_format(ts_with_exog_nan_begin_numpy, output_dims, segment_shape_expected, window_shape_expected):
    model = TSTCCEmbeddingModel(input_dims=3, output_dims=output_dims, batch_size=3)
    segment_embeddings = model.encode_segment(ts_with_exog_nan_begin_numpy)
    window_embeddings = model.encode_window(ts_with_exog_nan_begin_numpy)
    assert segment_embeddings.shape == segment_shape_expected
    assert window_embeddings.shape == window_shape_expected


def test_encode_pre_fitted(ts_with_exog_nan_begin_numpy, tmp_path):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=3)
    model.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)
    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)

    model_loaded = TSTCCEmbeddingModel.load(path=path)

    np.testing.assert_array_equal(
        model.encode_window(ts_with_exog_nan_begin_numpy), model_loaded.encode_window(ts_with_exog_nan_begin_numpy)
    )
    np.testing.assert_array_equal(
        model.encode_segment(ts_with_exog_nan_begin_numpy), model_loaded.encode_segment(ts_with_exog_nan_begin_numpy)
    )


def test_not_freeze_fit(ts_with_exog_nan_begin_numpy, tmp_path):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=3)
    model.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)
    model.freeze(is_freezed=False)
    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)

    model_loaded = TSTCCEmbeddingModel.load(path=path)
    model_loaded.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)

    assert model_loaded.is_freezed is False
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            model.encode_window(ts_with_exog_nan_begin_numpy), model_loaded.encode_window(ts_with_exog_nan_begin_numpy)
        )
        np.testing.assert_array_equal(
            model.encode_segment(ts_with_exog_nan_begin_numpy),
            model_loaded.encode_segment(ts_with_exog_nan_begin_numpy),
        )


def test_freeze_fit(ts_with_exog_nan_begin_numpy, tmp_path):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=3)
    model.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)
    model.freeze(is_freezed=True)
    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)

    model_loaded = TSTCCEmbeddingModel.load(path=path)
    model_loaded.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)

    assert model_loaded.is_freezed is True
    np.testing.assert_array_equal(
        model.encode_window(ts_with_exog_nan_begin_numpy), model_loaded.encode_window(ts_with_exog_nan_begin_numpy)
    )
    np.testing.assert_array_equal(
        model.encode_segment(ts_with_exog_nan_begin_numpy), model_loaded.encode_segment(ts_with_exog_nan_begin_numpy)
    )


@pytest.mark.parametrize(
    "data, input_dim",
    [("ts_with_exog_nan_begin_numpy", 3), ("ts_with_exog_nan_middle_numpy", 2), ("ts_with_exog_nan_end_numpy", 1)],
)
def test_encode_not_contains_nan(data, input_dim, request):
    data = request.getfixturevalue(data)
    model = TSTCCEmbeddingModel(input_dims=input_dim, batch_size=2)
    model.fit(data, n_epochs=1)
    encoded_segment = model.encode_segment(data)
    encoded_window = model.encode_window(data)

    assert np.isnan(encoded_segment).sum() == 0
    assert np.isnan(encoded_window).sum() == 0


def test_failed_batch_size_1(ts_with_exog_nan_begin_numpy):
    model = TSTCCEmbeddingModel(input_dims=3, batch_size=1)
    with pytest.raises(ValueError):
        model.fit(ts_with_exog_nan_begin_numpy, n_epochs=1)
