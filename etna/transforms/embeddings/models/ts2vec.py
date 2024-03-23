import pathlib
import tempfile
import zipfile
from typing import Optional
from typing import Union

import numpy as np
from typing_extensions import Literal

from etna.libs.ts2vec import TS2Vec
from etna.transforms.embeddings.models import BaseEmbeddingModel


class TS2VecEmbeddingModel(BaseEmbeddingModel):
    """TS2Vec embedding model.

    If there are NaNs in series, embeddings will not contain NaNs.

    For more details read the
    `paper <https://arxiv.org/abs/2106.10466>`_.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        device: Union[Literal["cpu"], Literal["gpu"]] = "cpu",
        lr: float = 0.001,
        batch_size: int = 16,
        max_train_length: Optional[int] = None,
        temporal_unit: int = 0,
    ):
        """Init TS2VecEmbeddingModel.

        Parameters
        ----------
        input_dims:
            The input dimension. For a univariate time series, this should be set to 1.
        output_dims:
            The representation dimension.
        hidden_dims:
            The hidden dimension of the encoder.
        depth:
            The number of hidden residual blocks in the encoder.
        device:
            The device used for training and inference.
        lr:
            The learning rate.
        batch_size:
            The batch size.
        max_train_length:
            The maximum allowed sequence length for training. For sequence with a length greater than ``max_train_length``,
            it would be cropped into some sequences, each of which has a length less than ``max_train_length``.
        temporal_unit:
            The minimum unit to perform temporal contrast. When training on a very long sequence,
            this param helps to reduce the cost of time and memory.
        Notes
        -----
        In case of long series to reduce memory consumption it is recommended to use max_train_length parameter or manually break the series into smaller subseries.
        """
        super().__init__(output_dims=output_dims)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self.device = device
        self.lr = lr
        self.batch_size = batch_size

        self.embedding_model = TS2Vec(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            max_train_length=self.max_train_length,
            temporal_unit=self.temporal_unit,
            device=self.device,
            lr=self.lr,
            batch_size=self.batch_size,
        )

        self._is_freezed: bool = False

    @property
    def is_freezed(self):
        """Return whether to skip training during ``fit``."""
        return self._is_freezed

    def freeze(self, is_freezed: bool = True):
        """Enable or disable skipping training in ``fit``.

        Parameters
        ----------
        is_freezed:
            whether to skip training during ``fit``.
        """
        self._is_freezed = is_freezed

    def fit(
        self,
        x: np.ndarray,
        n_epochs: Optional[int] = None,
        n_iters: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> "TS2VecEmbeddingModel":
        """Fit TS2Vec embedding model.

        Parameters
        ----------
        x:
            data with shapes (n_segments, n_timestamps, input_dims).
        n_epochs:
            The number of epochs. When this reaches, the training stops.
        n_iters:
            The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified,
            a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
        verbose:
            Whether to print the training loss after each epoch.
        """
        if not self._is_freezed:
            self.embedding_model.fit(train_data=x, n_epochs=n_epochs, n_iters=n_iters, verbose=verbose)
        return self

    def encode_segment(
        self,
        x: np.ndarray,
        mask: Union[
            Literal["binomial"], Literal["continuous"], Literal["all_true"], Literal["all_false"], Literal["mask_last"]
        ] = "all_true",
        sliding_length: Optional[int] = None,
        sliding_padding: int = 0,
    ) -> np.ndarray:
        """Create embeddings of the whole series.

        Parameters
        ----------
        x:
            data with shapes (n_segments, n_timestamps, input_dims).
        mask:
            the mask used by encoder on the test phase can be specified with this parameter. The possible options are:
            * 'binomial' - mask timestamp with probability 0.5 (default one, used in the paper). It is used on the training phase.
            * 'continuous' - mask random windows of timestamps
            * 'all_true' - mask none of the timestamps
            * 'all_false' - mask all timestamps
            * 'mask_last' - mask last timestamp
        sliding_length:
            the length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
        sliding_padding:
            contextual data length used for inference every sliding windows.

        Returns
        -------
        :
            array with embeddings of shape (n_segments, output_dim)
        """
        embeddings = self.embedding_model.encode(  # (n_segments, output_dim)
            data=x,
            mask=mask,
            encoding_window="full_series",
            causal=False,
            sliding_length=sliding_length,
            sliding_padding=sliding_padding,
            batch_size=self.batch_size,
        )

        return embeddings

    def encode_window(
        self,
        x: np.ndarray,
        mask: Union[
            Literal["binomial"], Literal["continuous"], Literal["all_true"], Literal["all_false"], Literal["mask_last"]
        ] = "all_true",
        sliding_length: Optional[int] = None,
        sliding_padding: int = 0,
        encoding_window: Optional[Union[Literal["multiscale"], int]] = None,
    ) -> np.ndarray:
        """Create embeddings of each series timestamp.

        Parameters
        ----------
        x:
            data with shapes (n_segments, n_timestamps, input_dims).
        mask:
            the mask used by encoder on the test phase can be specified with this parameter. The possible options are:
            * 'binomial' - mask timestamp with probability 0.5 (default one, used in the paper). It is used on the training phase.
            * 'continuous' - mask random windows of timestamps
            * 'all_true' - mask none of the timestamps
            * 'all_false' - mask all timestamps
            * 'mask_last' - mask last timestamp
        sliding_length:
            the length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
        sliding_padding:
            the contextual data length used for inference every sliding windows.
        encoding_window:
            when this param is specified, the computed representation would the max pooling over this window. The possible options are:
            * 'multiscale'
            * integer specifying the pooling kernel size.
            This param will be ignored when encoding full series

        Returns
        -------
        :
            array with embeddings of shape (n_segments, n_timestamps, output_dim)
        """
        embeddings = self.embedding_model.encode(  # (n_segments, n_timestamps, output_dim)
            data=x,
            mask=mask,
            encoding_window=encoding_window,
            causal=True,
            sliding_length=sliding_length,
            sliding_padding=sliding_padding,
            batch_size=self.batch_size,
        )
        return embeddings

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

                # save model separately
                model_save_path = temp_dir / "model.pt"
                self.embedding_model.save(fn=str(model_save_path))
                archive.write(model_save_path, "model.zip")

    @classmethod
    def load(cls, path: pathlib.Path) -> "TS2VecEmbeddingModel":
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
        obj: TS2VecEmbeddingModel = super().load(path=path)
        obj.embedding_model = TS2Vec(
            input_dims=obj.input_dims,
            output_dims=obj.output_dims,
            hidden_dims=obj.hidden_dims,
            depth=obj.depth,
            max_train_length=obj.max_train_length,
            temporal_unit=obj.temporal_unit,
            device=obj.device,
            lr=obj.lr,
            batch_size=obj.batch_size,
        )

        with zipfile.ZipFile(path, "r") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                archive.extractall(temp_dir)

                model_path = temp_dir / "model.zip"
                obj.embedding_model.load(fn=str(model_path))

        return obj
