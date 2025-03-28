import pathlib
import tempfile
import zipfile
from typing import Optional

from typing_extensions import Self

from etna.core import SaveMixin
from etna.core import load
from etna.datasets import TSDataset
from etna.pipeline import BasePipeline


class SavePredictionIntervalsMixin(SaveMixin):
    """Implementation of ``AbstractSaveable`` abstract class for prediction intervals with pipelines inside.

    It saves object to the zip archive with 3 entities:

    * metadata.json: contains library version and class name.

    * object.pkl: pickled without pipeline and ts.

    * pipeline.zip: pipeline archive, saved with its own method.
    """

    def save(self, path: pathlib.Path):
        """Save the object.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        self.pipeline: BasePipeline

        self._save(path=path, skip_attributes=["pipeline"])

        with zipfile.ZipFile(path, "a") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                # save pipeline separately and add to the archive
                pipeline_save_path = temp_dir / "pipeline.zip"
                self.pipeline.save(path=pipeline_save_path)

                archive.write(pipeline_save_path, "pipeline.zip")

    @classmethod
    def load(cls, path: pathlib.Path, ts: Optional[TSDataset] = None) -> Self:
        """Load an object.

        Warning
        -------
        This method uses :py:mod:`dill` module which is not secure.
        It is possible to construct malicious data which will execute arbitrary code during loading.
        Never load data that could have come from an untrusted source, or that could have been tampered with.

        Parameters
        ----------
        path:
            Path to load object from.
        ts:
            TSDataset to set into loaded pipeline.

        Returns
        -------
        :
            Loaded object.
        """
        obj = super().load(path=path)

        with zipfile.ZipFile(path, "r") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                archive.extractall(temp_dir)

                # load pipeline and add to the object
                pipeline_path = temp_dir / "pipeline.zip"
                obj.pipeline = load(path=pipeline_path, ts=ts)

        return obj
