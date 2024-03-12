import pathlib

import numpy as np
import pytest

from etna.transforms.embeddings.models import TS2VecEmbeddingModel


@pytest.fixture
def ts_with_exog_nan_begin_numpy(ts_with_exog_nan_begin) -> np.ndarray:
    n_features = 3
    df = ts_with_exog_nan_begin.to_pandas()
    n_timestamps = len(df.index)
    n_segments = df.columns.get_level_values("segment").nunique()
    x = df.values.reshape((n_timestamps, n_segments, n_features)).transpose(1, 0, 2)
    return x


@pytest.fixture
def ts_with_exog_nan_middle_numpy(ts_with_exog_nan_middle) -> np.ndarray:
    n_features = 2
    df = ts_with_exog_nan_middle.to_pandas()
    n_timestamps = len(df.index)
    n_segments = df.columns.get_level_values("segment").nunique()
    x = df.values.reshape((n_timestamps, n_segments, n_features)).transpose(1, 0, 2)
    return x


@pytest.fixture
def ts_with_exog_nan_end_numpy(ts_with_exog_nan_end) -> np.ndarray:
    n_features = 1
    df = ts_with_exog_nan_end.to_pandas()
    n_timestamps = len(df.index)
    n_segments = df.columns.get_level_values("segment").nunique()
    x = df.values.reshape((n_timestamps, n_segments, n_features)).transpose(1, 0, 2)
    return x


@pytest.mark.smoke
def test_fit(ts_with_exog_nan_begin_numpy):
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    model.fit(ts_with_exog_nan_begin_numpy)


@pytest.mark.smoke
def test_encode_segment(ts_with_exog_nan_begin_numpy):
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_segment(ts_with_exog_nan_begin_numpy)


@pytest.mark.smoke
def test_encode_window(ts_with_exog_nan_begin_numpy):
    model = TS2VecEmbeddingModel(input_dims=3)
    model.encode_window(ts_with_exog_nan_begin_numpy)


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
    "output_dims, segment_shape_expected, window_shape_expected", [(2, (5, 2), (5, 10, 2)), (3, (5, 3), (5, 10, 3))]
)
def test_encode_format(ts_with_exog_nan_begin_numpy, output_dims, segment_shape_expected, window_shape_expected):
    model = TS2VecEmbeddingModel(input_dims=3, output_dims=output_dims)
    segment_embeddings = model.encode_segment(ts_with_exog_nan_begin_numpy)
    window_embeddings = model.encode_window(ts_with_exog_nan_begin_numpy)
    assert segment_embeddings.shape == segment_shape_expected
    assert window_embeddings.shape == window_shape_expected


def test_encode_pre_fitted(ts_with_exog_nan_begin_numpy, tmp_path):
    model = TS2VecEmbeddingModel(input_dims=3, n_epochs=1)
    assert model._is_fitted is False

    model.fit(ts_with_exog_nan_begin_numpy)
    assert model._is_fitted is True

    path = pathlib.Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    model_loaded = TS2VecEmbeddingModel.load(path=path)
    assert model_loaded._is_fitted is True

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
    model = TS2VecEmbeddingModel(input_dims=input_dim, n_epochs=1)
    model.fit(data)
    encoded_segment = model.encode_segment(data)
    encoded_window = model.encode_window(data)

    assert np.isnan(encoded_segment).sum() == 0
    assert np.isnan(encoded_window).sum() == 0
