from abc import abstractmethod

import numpy as np

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
    def fit(self, x: np.ndarray) -> "BaseEmbeddingModel":
        """Fit the embedding model."""
        pass

    @abstractmethod
    def encode_segment(self, x: np.ndarray) -> np.ndarray:
        """Create embeddings of the input data."""
        pass

    @abstractmethod
    def encode_window(self, x: np.ndarray) -> np.ndarray:
        """Create embeddings of the input data."""
        pass
