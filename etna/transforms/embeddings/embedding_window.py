import pathlib
import tempfile
import zipfile
from typing import List

import pandas as pd

from etna.core import load
from etna.transforms.base import IrreversibleTransform
from etna.transforms.embeddings.models import BaseEmbeddingModel


class EmbeddingWindowTransform(IrreversibleTransform):
    """Create the embedding features for each timestamp using embedding model."""

    def __init__(self, in_columns: List[str], embedding_model: BaseEmbeddingModel, out_column: str = "embedding_window"):
        """Init EmbeddingWindowTransform.

        Parameters
        ----------
        in_columns:
            Columns to use for creating embeddings
        embedding_model:
            Model to create the embeddings
        out_column:
            Prefix for output columns, the output columns format is '{out_column}_{i}'
        """
        super().__init__(required_features=in_columns)
        self.in_columns = in_columns
        self.embedding_model = embedding_model
        self.out_column = out_column

    def _get_out_columns(self) -> List[str]:
        """Create the output columns names."""
        return [f"{self.out_column}_{i}" for i in range(self.embedding_model.output_dims)]

    def _fit(self, df: pd.DataFrame):
        """Fit transform."""
        self.embedding_model.fit(df.values)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create embedding features."""
        segments = df.columns.get_level_values("segment").unique()
        embeddings = self.embedding_model.encode_window(df.values)  # (n_timestamps, n_segments * output_dim)

        df_encoded = pd.DataFrame(
            embeddings, columns=pd.MultiIndex.from_product([segments, self._get_out_columns()]), index=df.index
        )
        df = pd.concat([df, df_encoded], axis=1)
        df = df.sort_index(axis=1)
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def save(self, path: pathlib.Path):
        """Save the object.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        self._save(path=path, skip_attributes=["embedding_model"])

        # Save embedding_model
        with zipfile.ZipFile(path, "a") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                model_save_path = temp_dir / "model.zip"
                self.embedding_model.save(path=model_save_path)
                archive.write(model_save_path, "model.zip")

    @classmethod
    def load(cls, path: pathlib.Path) -> "EmbeddingWindowTransform":
        """Load an object.

        Parameters
        ----------
        path:
            Path to load object from.

        Returns
        -------
        :
            Loaded object.
        """
        # Load transform embedding_model
        obj: EmbeddingWindowTransform = super().load(path=path)

        # Load embedding_model
        with zipfile.ZipFile(path, "r") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                archive.extractall(temp_dir)

                model_path = temp_dir / "model.zip"
                obj.embedding_model = load(path=model_path)

        return obj
