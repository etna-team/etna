from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from etna import SETTINGS
from etna.distributions import BaseDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.models.base import DeepBaseModel
from etna.models.base import DeepBaseNet

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn

    from etna.models.nn.utils import MultiEmbedding


class MLPBatch(TypedDict):
    """Batch specification for MLP."""

    decoder_real: "torch.Tensor"
    decoder_target: "torch.Tensor"
    decoder_categorical: Dict[str, "torch.Tensor"]
    segment: "torch.Tensor"


class MLPNet(DeepBaseNet):
    """MLP model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int],
        embedding_sizes: Dict[str, Tuple[int, int]],
        lr: float,
        loss: "torch.nn.Module",
        optimizer_params: Optional[dict],
    ) -> None:
        """Init MLP model.

        Parameters
        ----------
        input_size:
            size of the input feature space: target plus extra features
        hidden_size:
            list of sizes of the hidden states
        embedding_sizes:
            dictionary mapping categorical feature name to tuple of number of categorical classes and embedding size
        lr:
            learning rate
        loss:
            loss function
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_sizes = embedding_sizes
        self.lr = lr
        self.loss = loss
        self.optimizer_params = {} if optimizer_params is None else optimizer_params
        self.cat_size = sum([dim for (_, dim) in self.embedding_sizes.values()])
        self.embedding: Optional[MultiEmbedding] = None
        if self.embedding_sizes:
            self.embedding = MultiEmbedding(
                embedding_sizes=self.embedding_sizes,
            )
        layers = [nn.Linear(in_features=input_size + self.cat_size, out_features=hidden_size[0]), nn.ReLU()]
        for i in range(1, len(hidden_size)):
            layers.append(nn.Linear(in_features=hidden_size[i - 1], out_features=hidden_size[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_size[-1], out_features=1))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _validate_batch(batch: MLPBatch):
        if batch["decoder_real"].isnan().sum().item():
            raise ValueError("There are NaNs in features, this model can't work with them!")

    def forward(self, batch: MLPBatch):  # type: ignore
        """Forward pass.

        Parameters
        ----------
        batch:
            batch of data
        Returns
        -------
        :
            forecast
        """
        self._validate_batch(batch)
        decoder_real = batch["decoder_real"].float()
        decoder_categorical = batch["decoder_categorical"]  # each (batch_size, decoder_length, 1)

        decoder_embeddings = (
            self.embedding(decoder_categorical)
            if self.embedding is not None
            else torch.zeros((decoder_real.shape[0], decoder_real.shape[1], 0), device=decoder_real.device)
        )

        decoder_values = torch.concat((decoder_real, decoder_embeddings), dim=2)

        return self.mlp(decoder_values)

    def step(self, batch: MLPBatch, *args, **kwargs):  # type: ignore
        """Step for loss computation for training or validation.

        Parameters
        ----------
        batch:
            batch of data
        Returns
        -------
        :
            loss, true_target, prediction_target
        """
        self._validate_batch(batch)
        decoder_real = batch["decoder_real"].float()
        decoder_categorical = batch["decoder_categorical"]  # each (batch_size, decoder_length, 1)
        decoder_target = batch["decoder_target"].float()

        decoder_embeddings = (
            self.embedding(decoder_categorical)
            if self.embedding is not None
            else torch.zeros((decoder_real.shape[0], decoder_real.shape[1], 0), device=decoder_real.device)
        )

        decoder_values = torch.concat((decoder_real, decoder_embeddings), dim=2)
        output = self.mlp(decoder_values)
        loss = self.loss(output, decoder_target)
        return loss, decoder_target, output

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterable[dict]:
        """Make samples from segment DataFrame."""
        values_real = (
            df.drop(["target", "segment", "timestamp"] + list(self.embedding_sizes.keys()), axis=1)
            .select_dtypes(include=[np.number])
            .values.astype(np.float32)
        )

        # Categories that were not seen during `fit` will be filled with new category
        for feature in self.embedding_sizes:
            df[feature] = df[feature].astype(np.float32).fillna(self.embedding_sizes[feature][0])

        # Columns in `values_categorical` are in the same order as in `embedding_sizes`
        values_categorical = df[self.embedding_sizes.keys()].values.T

        values_target = df["target"].values
        segment = df["segment"].values[0]

        def _make(
            values_target: np.ndarray,
            values_real: np.ndarray,
            values_categorical: np.ndarray,
            segment: str,
            start_idx: int,
            decoder_length: int,
        ) -> Optional[dict]:

            sample: Dict[str, Any] = {
                "decoder_real": list(),
                "decoder_categorical": dict(),
                "decoder_target": list(),
                "segment": None,
            }
            total_length = len(df["target"])
            total_sample_length = decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            sample["decoder_real"] = values_real[start_idx : start_idx + decoder_length]

            for index, feature in enumerate(self.embedding_sizes.keys()):
                sample["decoder_categorical"][feature] = values_categorical[index][
                    start_idx + encoder_length : start_idx + total_sample_length
                ].reshape(-1, 1)

            sample["decoder_target"] = values_target[start_idx : start_idx + decoder_length].reshape(-1, 1)
            sample["segment"] = segment
            return sample

        start_idx = 0
        while True:
            batch = _make(
                values_target=values_target,
                values_real=values_real,
                values_categorical=values_categorical,
                segment=segment,
                start_idx=start_idx,
                decoder_length=decoder_length,
            )
            if batch is None:
                break
            yield batch
            start_idx += decoder_length
        if start_idx < len(df):
            resid_length = len(df) - decoder_length
            batch = _make(
                values_target=values_target,
                values_real=values_real,
                values_categorical=values_categorical,
                segment=segment,
                start_idx=resid_length,
                decoder_length=decoder_length,
            )
            if batch is not None:
                yield batch

    def configure_optimizers(self):
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        return optimizer


class MLPModel(DeepBaseModel):
    """MLPModel.

    Model needs label encoded inputs for categorical features, for that purposes use :py:class:`~etna.transforms.LabelEncoderTransform`.
    Feature values that weren't seen during ``fit`` should be set to NaN, to get this behaviour use encoder with *strategy="none"*.

    If there are numeric columns that are passed to ``embedding_sizes`` parameter, they will be considered only as categorical features.

    Note
    ----
    This model requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        input_size: int,
        decoder_length: int,
        hidden_size: List,
        embedding_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        encoder_length: int = 0,
        lr: float = 1e-3,
        loss: Optional["torch.nn.Module"] = None,
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        """Init MLP model.

        Parameters
        ----------
        input_size:
            size of the input numeric feature space without target
        decoder_length:
            decoder length
        hidden_size:
            List of sizes of the hidden states
        embedding_sizes:
            dictionary mapping categorical feature name to tuple of number of categorical classes and embedding size
        encoder_length:
            encoder length
        lr:
            learning rate
        loss:
            loss function, MSELoss by default
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch ligthning trainer parameters (api reference :py:class:`lightning.pytorch.trainer.trainer.Trainer`)
        train_dataloader_params:
            parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`)
        test_dataloader_params:
            parameters for test dataloader
        val_dataloader_params:
            parameters for validation dataloader
        split_params:
            dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training
                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting
                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_sizes = embedding_sizes
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params
        super().__init__(
            net=MLPNet(
                input_size=input_size,
                hidden_size=hidden_size,
                embedding_sizes=embedding_sizes if embedding_sizes is not None else {},
                lr=lr,
                loss=nn.MSELoss() if loss is None else loss,
                optimizer_params=optimizer_params,
            ),
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``lr``, ``hidden_size.i`` where i from 0 to ``len(hidden_size) - 1``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        grid: Dict[str, BaseDistribution] = {}

        for i in range(len(self.hidden_size)):
            key = f"hidden_size.{i}"
            value = IntDistribution(low=4, high=64, step=4)
            grid[key] = value

        grid["lr"] = FloatDistribution(low=1e-5, high=1e-2, log=True)
        return grid
