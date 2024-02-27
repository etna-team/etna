from abc import abstractmethod

import numpy as np
import pandas as pd

from etna.core import BaseMixin
from etna.core import SaveMixin


class BaseEmbeddingModel(BaseMixin, SaveMixin):
    """Base class for embedding models."""

    def __init__(self, output_dims: int):
        """Init BaseEmbeddingModel.

        Parameters
        ----------
        output_dims:
            Dimension of the output embeddings
        """
        super().__init__()
        self.output_dims = output_dims

    @abstractmethod
    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for the embedding model."""
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseEmbeddingModel":
        """Fit the embedding model."""
        pass

    @abstractmethod
    def encode_segment(self, df: pd.DataFrame) -> np.ndarray:
        """Create embeddings of the input data."""
        pass

    @abstractmethod
    def encode_window(self, df: pd.DataFrame) -> np.ndarray:
        """Create embeddings of the input data."""
        pass
