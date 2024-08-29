import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from loguru import logger as _logger

from etna.loggers import ConsoleLogger
from etna.loggers import tslogger
from etna.transforms.embeddings.models import TSTCCEmbeddingModel
from tests.test_transforms.test_embeddings.test_models.utils import check_logged_loss


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

    path = Path(tmp_path) / "tmp.zip"
    model.save(path=path)


@pytest.mark.smoke
def test_load(tmp_path):
    model = TSTCCEmbeddingModel(input_dims=3)

    path = Path(tmp_path) / "tmp.zip"
    model.save(path=path)
    TSTCCEmbeddingModel.load(path=path)


@pytest.mark.smoke
def test_list_models():
    TSTCCEmbeddingModel.list_models()


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
    path = Path(tmp_path) / "tmp.zip"
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
    path = Path(tmp_path) / "tmp.zip"
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
    path = Path(tmp_path) / "tmp.zip"
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


@pytest.mark.parametrize("verbose, n_epochs, n_lines_expected", [(True, 1, 1), (False, 1, 0)])
def test_logged_loss(ts_with_exog_nan_begin_numpy, verbose, n_epochs, n_lines_expected):
    """Check logging loss during training."""
    model = TSTCCEmbeddingModel(input_dims=3)
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    model.fit(ts_with_exog_nan_begin_numpy, n_epochs=n_epochs, verbose=verbose)
    check_logged_loss(log_file=file.name, n_lines_expected=n_lines_expected)
    tslogger.remove(idx)


def test_correct_list_models():
    assert TSTCCEmbeddingModel.list_models() == ["tstcc_medium"]


@pytest.mark.parametrize("model_name", ["tstcc_medium"])
def test_load_pretrained_model_default_path(model_name):
    path = Path.home() / ".etna" / "embeddings" / "tstcc" / f"{model_name}.zip"
    _ = TSTCCEmbeddingModel.load(path=path, model_name=model_name)
    assert os.path.isfile(path)


@pytest.mark.parametrize("model_name", ["tstcc_medium"])
def test_load_pretrained_model_exact_path(model_name, tmp_path):
    path = Path(tmp_path) / "tmp.zip"
    _ = TSTCCEmbeddingModel.load(path=path, model_name=model_name)
    assert os.path.isfile(path)


def test_load_unknown_pretrained_model():
    model_name = "unknown_model"
    with pytest.raises(
        NotImplementedError,
        match=f"Model {model_name} is not available. To get list of available models use `list_models` method.",
    ):
        TSTCCEmbeddingModel.load(model_name=model_name)


def test_load_set_none_parameters():
    with pytest.raises(
        ValueError, match="Both path and model_name are not specified. At least one parameter should be specified."
    ):
        TSTCCEmbeddingModel.load()


def test_warning_existing_path(tmp_path):
    model = TSTCCEmbeddingModel(input_dims=1)
    path = Path(tmp_path) / "tmp.zip"
    model.save(path)

    model_name = "tstcc_medium"
    with pytest.warns(
        UserWarning,
        match=f"Path {path} already exists. Model {model_name} will not be downloaded. Loading existing local model.",
    ):
        TSTCCEmbeddingModel.load(path=path, model_name=model_name)
