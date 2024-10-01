import os
import pathlib
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from urllib import request

import numpy as np

from etna import SETTINGS
from etna.transforms.embeddings.models import BaseEmbeddingModel

if SETTINGS.torch_required:
    from etna.libs.tstcc import TSTCC

_DOWNLOAD_PATH = Path.home() / ".etna" / "embeddings" / "tstcc"


class TSTCCEmbeddingModel(BaseEmbeddingModel):
    """TSTCC embedding model.

    If there are NaNs in series, embeddings will not contain NaNs.

    Each following calling of ``fit`` method continues the learning of the same model.

    Using custom `output_dims`, set it to a value > 3 to have the loss calculated correctly.

    For more details read the
    `paper <https://arxiv.org/abs/2106.14112>`_.

    Notes
    -----

    - This model cannot be fitted with `batch_size=1`. So, it cannot be fitted on a dataset with 1 segment.
    - Model's weights are transferred to cpu during loading.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 32,
        tc_hidden_dim: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.35,
        timesteps: int = 7,
        heads: int = 1,
        depth: int = 4,
        jitter_scale_ratio: float = 1.1,
        max_seg: int = 4,
        jitter_ratio: float = 0.8,
        use_cosine_similarity: bool = True,
        n_seq_steps: int = 0,
        device: Literal["cpu", "cuda"] = "cpu",
        batch_size: int = 16,
        num_workers: int = 0,
        is_freezed: bool = False,
    ):
        """Init TSTCCEmbeddingModel.

        Parameters
        ----------
        input_dims:
            The input dimension. For a univariate time series, this should be set to 1.
        output_dims:
            The representation dimension.
        tc_hidden_dim:
            The output dimension after temporal_contr_model.
        kernel_size:
            Kernel size of first convolution in encoder.
        dropout:
            Dropout rate in first convolution block in encoder.
        timesteps:
            The number of timestamps to predict in temporal contrasting model.
        heads:
            Number of heads in attention block in temporal contrasting model. Parameter output_dims must be a multiple
            of the number of heads.
        depth:
            Depth in attention block in temporal contrasting model.
        n_seq_steps:
            Max context size in temporal contrasting model.
        jitter_scale_ratio:
            Jitter ratio in weak augmentation.
        max_seg:
            Number of segments in strong augmentation.
        jitter_ratio:
            Jitter ratio in strong augmentation.
        use_cosine_similarity:
            If True NTXentLoss uses cosine similarity, if False NTXentLoss uses dot product.
        device:
            The device used for training and inference. To swap device, change this attribute.
        batch_size:
            The batch size (number of segments in a batch). To swap batch_size, change this attribute.
        num_workers:
            How many subprocesses to use for data loading. See (api reference :py:class:`torch.utils.data.DataLoader`). To swap num_workers, change this attribute.
        is_freezed:
            Whether to ``freeze`` model in constructor or not. For more details see ``freeze`` method.
        """
        super().__init__(output_dims=output_dims)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tc_hidden_dim = tc_hidden_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.timesteps = timesteps
        self.heads = heads
        self.depth = depth
        self.n_seq_steps = n_seq_steps

        self.jitter_scale_ratio = jitter_scale_ratio
        self.max_seg = max_seg
        self.jitter_ratio = jitter_ratio

        self.use_cosine_similarity = use_cosine_similarity

        self.batch_size = batch_size

        self.device = device
        self.num_workers = num_workers

        self.embedding_model = TSTCC(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            timesteps=self.timesteps,
            tc_hidden_dim=self.tc_hidden_dim,
            heads=self.heads,
            depth=self.depth,
            n_seq_steps=self.n_seq_steps,
            jitter_scale_ratio=self.jitter_scale_ratio,
            max_seg=self.max_seg,
            jitter_ratio=self.jitter_ratio,
            use_cosine_similarity=self.use_cosine_similarity,
        )
        self._is_freezed = is_freezed

        if self._is_freezed:
            self.freeze()

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
        n_epochs: int = 40,
        lr: float = 0.001,
        temperature: float = 0.2,
        lambda1: float = 1,
        lambda2: float = 0.7,
        verbose: bool = False,
    ) -> "TSTCCEmbeddingModel":
        """Fit TSTCC embedding model.

        Parameters
        ----------
        x:
            data with shapes (n_segments, n_timestamps, input_dims).
        n_epochs:
            The number of epochs. When this reaches, the training stops.
        lr:
            The learning rate.
        temperature:
            Temperature in NTXentLoss.
        lambda1:
            The relative weight of the first item in the loss (temporal contrasting loss).
        lambda2:
            The relative weight of the second item in the loss (contextual contrasting loss).
        verbose:
            Whether to print the training loss after each epoch.
        """
        if not self._is_freezed:
            self.embedding_model.fit(
                train_data=x,
                n_epochs=n_epochs,
                lr=lr,
                temperature=temperature,
                lambda1=lambda1,
                lambda2=lambda2,
                verbose=verbose,
                device=self.device,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
            )
        return self

    def encode_segment(self, x: np.ndarray) -> np.ndarray:
        """Create embeddings of the whole series.

        Parameters
        ----------
        x:
            data with shapes (n_segments, n_timestamps, input_dims).
        Returns
        -------
        :
            array with embeddings of shape (n_segments, output_dim)
        """
        embeddings = self.embedding_model.encode(
            data=x,
            encode_full_series=True,
            device=self.device,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )  # (n_segments, output_dim)

        return embeddings

    def encode_window(self, x: np.ndarray) -> np.ndarray:
        """Create embeddings of each series timestamp.

        Parameters
        ----------
        x:
            data with shapes (n_segments, n_timestamps, input_dims).

        Returns
        -------
        :
            array with embeddings of shape (n_segments, n_timestamps, output_dim)
        """
        embeddings = self.embedding_model.encode(
            data=x,
            encode_full_series=False,
            device=self.device,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )  # (n_segments, n_timestamps, output_dim)
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
    def load(cls, path: Optional[pathlib.Path] = None, model_name: Optional[str] = None) -> "TSTCCEmbeddingModel":
        """Load an object.

        Model's weights are transferred to cpu during loading.

        Parameters
        ----------
        path:
            Path to load object from.

            - if ``path`` is not None and ``model_name`` is None, load the local model from ``path``.
            - if ``path`` is None and ``model_name`` is not None, save the external ``model_name`` model to the etna folder in the home directory and load it. If ``path`` exists, external model will not be downloaded.
            - if ``path`` is not  None and ``model_name`` is not None, save the external ``model_name`` model to ``path`` and load it. If ``path`` exists, external model will not be downloaded.

        model_name:
            name of external model to load. To get list of available models use ``list_models`` method.

        Returns
        -------
        :
            Loaded object.

        Raises
        ------
        ValueError:
            If none of parameters ``path`` and ``model_name`` are set.
        NotImplementedError:
            If ``model_name`` isn't from list of available model names.
        """
        warnings.filterwarnings(
            "ignore",
            message="The object was saved under etna version 2.7.1 but running version is",
            category=UserWarning,
        )

        if model_name is not None:
            if path is None:
                path = _DOWNLOAD_PATH / f"{model_name}.zip"
            if os.path.exists(path):
                warnings.warn(
                    f"Path {path} already exists. Model {model_name} will not be downloaded. Loading existing local model."
                )
            else:
                Path(path).parent.mkdir(exist_ok=True, parents=True)

                if model_name in cls.list_models():
                    url = f"http://etna-github-prod.cdn-tinkoff.ru/embeddings/tstcc/{model_name}.zip"
                    request.urlretrieve(url=url, filename=path)
                else:
                    raise NotImplementedError(
                        f"Model {model_name} is not available. To get list of available models use `list_models` method."
                    )
        elif path is None and model_name is None:
            raise ValueError("Both path and model_name are not specified. At least one parameter should be specified.")

        obj: TSTCCEmbeddingModel = super().load(path=path)
        obj.embedding_model = TSTCC(
            input_dims=obj.input_dims,
            output_dims=obj.output_dims,
            kernel_size=obj.kernel_size,
            dropout=obj.dropout,
            timesteps=obj.timesteps,
            heads=obj.heads,
            depth=obj.depth,
            tc_hidden_dim=obj.tc_hidden_dim,
            n_seq_steps=obj.n_seq_steps,
            jitter_scale_ratio=obj.jitter_scale_ratio,
            max_seg=obj.max_seg,
            jitter_ratio=obj.jitter_ratio,
            use_cosine_similarity=obj.use_cosine_similarity,
        )

        with zipfile.ZipFile(path, "r") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                archive.extractall(temp_dir)
                model_path = temp_dir / "model.zip"
                obj.embedding_model.load(fn=str(model_path))

        return obj

    @staticmethod
    def list_models() -> List[str]:
        """
        Return a list of available pretrained models.

        Main information about available models:

        - tstcc_medium:

          - Number of parameters - 234k
          - Dimension of output embeddings - 16

        Returns
        -------
        :
            List of available pretrained models.
        """
        return ["tstcc_medium"]
