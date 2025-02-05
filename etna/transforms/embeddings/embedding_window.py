import pathlib
import tempfile
import zipfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.core import load
from etna.transforms.base import IrreversibleTransform
from etna.transforms.embeddings.models import BaseEmbeddingModel


class EmbeddingWindowTransform(IrreversibleTransform):
    """Create the embedding features for each timestamp using embedding model."""

    def __init__(
        self,
        in_columns: List[str],
        embedding_model: BaseEmbeddingModel,
        encoding_params: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None,
        out_column: str = "embedding_window",
    ):
        """Init EmbeddingWindowTransform.

        Parameters
        ----------
        in_columns:
            Columns to use for creating embeddings
        embedding_model:
            Model to create the embeddings
        encoding_params:
            Parameters to use during encoding. Parameters for corresponding models can be found at :ref:`embedding section <embeddings>`.
        training_params:
            Parameters to use during training. Parameters for corresponding models can be found at :ref:`embedding section <embeddings>`.
        out_column:
            Prefix for output columns, the output columns format is '{out_column}_{i}'
        """
        super().__init__(required_features=in_columns)
        self.in_columns = in_columns
        self.embedding_model = embedding_model
        self.encoding_params = encoding_params if encoding_params is not None else {}
        self.training_params = training_params if training_params is not None else {}
        self.out_column = out_column

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Reshape data into (n_segments, n_timestamps, input_dims)."""
        n_timestamps = len(df.index)
        n_segments = df.columns.get_level_values("segment").nunique()
        x = df.values.reshape((n_timestamps, n_segments, len(self.in_columns))).transpose(1, 0, 2)
        return x

    def _get_out_columns(self) -> List[str]:
        """Create the output columns names."""
        return [f"{self.out_column}_{i}" for i in range(self.embedding_model.output_dims)]

    def _fit(self, df: pd.DataFrame):
        """Fit transform."""
        x = self._prepare_data(df)
        self.embedding_model.fit(x, **self.training_params)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create embedding features."""
        segments = df.columns.get_level_values("segment").unique()
        n_timestamps = len(df.index)
        x = self._prepare_data(df)
        embeddings = self.embedding_model.encode_window(
            x=x, **self.encoding_params
        )  # (n_segments, n_timestamps, output_dim)
        embeddings = embeddings.transpose(1, 0, 2).reshape(n_timestamps, -1)  # (n_timestamps, n_segments * output_dim)

        df_encoded = pd.DataFrame(
            embeddings,
            columns=pd.MultiIndex.from_product([segments, self._get_out_columns()], names=df.columns.names),
            index=df.index,
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
