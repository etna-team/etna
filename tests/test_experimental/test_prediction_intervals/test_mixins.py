import pathlib
import pickle
import zipfile
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from etna.models import NaiveModel
from etna.pipeline import Pipeline
from tests.test_experimental.test_prediction_intervals.common import DummyPredictionIntervals


@pytest.mark.parametrize("expected_filenames", ({"metadata.json", "object.pkl", "pipeline.zip"},))
def test_save(naive_pipeline_with_transforms, example_tsds, tmp_path, expected_filenames):
    dummy = DummyPredictionIntervals(pipeline=naive_pipeline_with_transforms, width=4)

    path = pathlib.Path(tmp_path) / "dummy.zip"

    initial_dummy = deepcopy(dummy)
    dummy.save(path)

    with zipfile.ZipFile(path, "r") as archive:
        files = archive.namelist()
        assert set(files) == expected_filenames

        with archive.open("object.pkl", "r") as file:
            loaded_obj = pickle.load(file)
        assert loaded_obj.width == dummy.width

    # basic check that we didn't break dummy object itself
    assert dummy.width == initial_dummy.width
    assert pickle.dumps(dummy.ts) == pickle.dumps(initial_dummy.ts)
    assert pickle.dumps(dummy.pipeline.model) == pickle.dumps(initial_dummy.pipeline.model)
    assert pickle.dumps(dummy.pipeline.transforms) == pickle.dumps(initial_dummy.pipeline.transforms)


def test_load_file_not_found_error():
    non_existent_path = pathlib.Path("archive.zip")
    with pytest.raises(FileNotFoundError):
        DummyPredictionIntervals.load(non_existent_path)


def test_load_with_ts(naive_pipeline_with_transforms, example_tsds, recwarn, tmp_path):
    dummy = DummyPredictionIntervals(pipeline=naive_pipeline_with_transforms, width=4)

    path = pathlib.Path(tmp_path) / "dummy.zip"
    dummy.save(path)

    loaded_obj = DummyPredictionIntervals.load(path=path, ts=example_tsds)

    assert loaded_obj.width == dummy.width
    assert loaded_obj.ts is not dummy.ts
    pd.testing.assert_frame_equal(loaded_obj.ts.to_pandas(), example_tsds.to_pandas())
    assert isinstance(loaded_obj.pipeline, Pipeline)
    assert isinstance(loaded_obj.pipeline.model, NaiveModel)
    assert len(loaded_obj.pipeline.transforms) == 2
    assert len(recwarn) == 0


def test_load_without_ts(naive_pipeline_with_transforms, recwarn, tmp_path):
    dummy = DummyPredictionIntervals(pipeline=naive_pipeline_with_transforms, width=4)

    path = pathlib.Path(tmp_path) / "dummy.zip"
    dummy.save(path)

    loaded_obj = DummyPredictionIntervals.load(path=path)

    assert loaded_obj.width == dummy.width
    assert loaded_obj.ts is None
    assert isinstance(loaded_obj.pipeline, Pipeline)
    assert isinstance(loaded_obj.pipeline.model, NaiveModel)
    assert len(loaded_obj.pipeline.transforms) == 2
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "save_version, load_version", [((1, 5, 0), (2, 5, 0)), ((2, 5, 0), (1, 5, 0)), ((1, 5, 0), (1, 3, 0))]
)
@patch("etna.core.mixins.get_etna_version")
def test_save_mixin_load_warning(
    get_version_mock, naive_pipeline_with_transforms, save_version, load_version, tmp_path
):
    dummy = DummyPredictionIntervals(pipeline=naive_pipeline_with_transforms, width=4)
    path = pathlib.Path(tmp_path) / "dummy.zip"

    get_version_mock.return_value = save_version
    dummy.save(path)

    save_version_str = ".".join(map(str, save_version))
    load_version_str = ".".join(map(str, load_version))
    with pytest.warns(
        UserWarning,
        match=f"The object was saved under etna version {save_version_str} but running version is {load_version_str}",
    ):
        get_version_mock.return_value = load_version
        _ = DummyPredictionIntervals.load(path=path)
