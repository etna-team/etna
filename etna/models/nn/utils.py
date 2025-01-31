from typing import Dict
from typing import Tuple

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch import nn


class MultiEmbedding(nn.Module):
    """Class for obtaining the embeddings of the categorical features."""

    def __init__(self, embedding_sizes: Dict[str, Tuple[int, int]]):
        """
        Init MultiEmbedding.

        Parameters
        ----------
        embedding_sizes:
            dictionary mapping feature name to tuple of number of categorical classes and embedding size
        """
        super().__init__()

        self.embedding_sizes = embedding_sizes
        # We should add one more embedding for new categories that were not seen during model's `fit`
        self.embedding = nn.ModuleDict(
            {feature: nn.Embedding(n + 1, dim) for feature, (n, dim) in self.embedding_sizes.items()}
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            dictionary mapping feature name to feature values
        """
        return torch.concat([self.embedding[feature](x[feature].int().squeeze(2)) for feature in x.keys()], dim=2)
