from typing import Any
from typing import Dict
from typing import Iterator
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


class RNNBatch(TypedDict):
    """Batch specification for RNN."""

    encoder_real: "torch.Tensor"
    decoder_real: "torch.Tensor"
    encoder_categorical: Dict[str, "torch.Tensor"]
    decoder_categorical: Dict[str, "torch.Tensor"]
    encoder_target: "torch.Tensor"
    decoder_target: "torch.Tensor"
    segment: "torch.Tensor"


class RNNNet(DeepBaseNet):
    """RNN based Lightning module with LSTM cell."""

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        embedding_sizes: Dict[str, Tuple[int, int]],
        lr: float,
        loss: "torch.nn.Module",
        optimizer_params: Optional[dict],
    ) -> None:
        """Init RNN based on LSTM cell.

        Parameters
        ----------
        input_size:
            size of the input numeric feature space: target plus extra numeric features
        num_layers:
            number of layers
        hidden_size:
            size of the hidden state
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
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_sizes = embedding_sizes
        self.loss = torch.nn.MSELoss() if loss is None else loss
        self.cat_size = sum([dim for (_, dim) in self.embedding_sizes.values()])
        self.embedding: Optional[MultiEmbedding] = None
        if self.embedding_sizes:
            self.embedding = MultiEmbedding(
                embedding_sizes=self.embedding_sizes,
            )
        self.rnn = nn.LSTM(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            input_size=self.input_size + self.cat_size,
            batch_first=True,
        )
        self.projection = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.lr = lr
        self.optimizer_params = {} if optimizer_params is None else optimizer_params

    def forward(self, x: RNNBatch, *args, **kwargs):  # type: ignore
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data

        Returns
        -------
        :
            forecast with shape (batch_size, decoder_length, 1)
        """
        encoder_real = x["encoder_real"].float()  # (batch_size, encoder_length-1, input_size)
        decoder_real = x["decoder_real"].float()  # (batch_size, decoder_length, input_size)
        encoder_categorical = x["encoder_categorical"]  # each (batch_size, encoder_length-1, 1)
        decoder_categorical = x["decoder_categorical"]  # each (batch_size, decoder_length, 1)
        decoder_target = x["decoder_target"].float()  # (batch_size, decoder_length, 1)
        decoder_length = decoder_real.shape[1]

        encoder_embeddings = (
            self.embedding(encoder_categorical)
            if self.embedding is not None
            else torch.zeros((encoder_real.shape[0], encoder_real.shape[1], 0), device=encoder_real.device)
        )
        decoder_embeddings = (
            self.embedding(decoder_categorical)
            if self.embedding is not None
            else torch.zeros((decoder_real.shape[0], decoder_real.shape[1], 0), device=decoder_real.device)
        )

        encoder_values = torch.concat((encoder_real, encoder_embeddings), dim=2)
        decoder_values = torch.concat((decoder_real, decoder_embeddings), dim=2)

        output, (h_n, c_n) = self.rnn(encoder_values)
        forecast = torch.zeros_like(decoder_target)  # (batch_size, decoder_length, 1)

        for i in range(decoder_length - 1):
            output, (h_n, c_n) = self.rnn(decoder_values[:, i, None], (h_n, c_n))
            forecast_point = self.projection(output[:, -1]).flatten()
            forecast[:, i, 0] = forecast_point
            decoder_values[:, i + 1, 0] = forecast_point

        # Last point is computed out of the loop because `decoder_real[:, i + 1, 0]` would cause index error
        output, (_, _) = self.rnn(decoder_values[:, decoder_length - 1, None], (h_n, c_n))
        forecast_point = self.projection(output[:, -1]).flatten()
        forecast[:, decoder_length - 1, 0] = forecast_point
        return forecast

    def step(self, batch: RNNBatch, *args, **kwargs):  # type: ignore
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
        encoder_real = batch["encoder_real"].float()  # (batch_size, encoder_length-1, input_size)
        decoder_real = batch["decoder_real"].float()  # (batch_size, decoder_length, input_size)
        encoder_categorical = batch["encoder_categorical"]  # each (batch_size, encoder_length-1, 1)
        decoder_categorical = batch["decoder_categorical"]  # each (batch_size, decoder_length, 1)

        decoder_target = batch["decoder_target"].float()  # (batch_size, decoder_length, 1)

        decoder_length = decoder_real.shape[1]

        encoder_embeddings = (
            self.embedding(encoder_categorical)
            if self.embedding is not None
            else torch.zeros((encoder_real.shape[0], encoder_real.shape[1], 0), device=encoder_real.device)
        )
        decoder_embeddings = (
            self.embedding(decoder_categorical)
            if self.embedding is not None
            else torch.zeros((decoder_real.shape[0], decoder_real.shape[1], 0), device=decoder_real.device)
        )

        encoder_values = torch.concat((encoder_real, encoder_embeddings), dim=2)
        decoder_values = torch.concat((decoder_real, decoder_embeddings), dim=2)

        output, (_, _) = self.rnn(torch.cat((encoder_values, decoder_values), dim=1))

        target_prediction = output[:, -decoder_length:]
        target_prediction = self.projection(target_prediction)  # (batch_size, decoder_length, 1)

        loss = self.loss(target_prediction, decoder_target)
        return loss, decoder_target, target_prediction

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """Make samples from segment DataFrame."""
        values_real = (
            df.drop(["segment", "timestamp"] + list(self.embedding_sizes.keys()), axis=1)
            .select_dtypes(include=[np.number])
            .assign(target_shifted=df["target"].shift(1))
            .drop(["target"], axis=1)
            .pipe(lambda x: x[["target_shifted"] + [i for i in x.columns if i != "target_shifted"]])
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
            values_real: np.ndarray,
            values_categorical: np.ndarray,
            values_target: np.ndarray,
            segment: str,
            start_idx: int,
            encoder_length: int,
            decoder_length: int,
        ) -> Optional[dict]:

            sample: Dict[str, Any] = {
                "encoder_real": list(),
                "decoder_real": list(),
                "encoder_categorical": dict(),
                "decoder_categorical": dict(),
                "encoder_target": list(),
                "decoder_target": list(),
                "segment": None,
            }
            total_length = len(values_target)
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            # Get shifted target and concatenate it with real values features
            sample["decoder_real"] = values_real[start_idx + encoder_length : start_idx + total_sample_length]

            # Get shifted target and concatenate it with real values features
            sample["encoder_real"] = values_real[start_idx : start_idx + encoder_length]
            sample["encoder_real"] = sample["encoder_real"][1:]

            for index, feature in enumerate(self.embedding_sizes.keys()):
                sample["encoder_categorical"][feature] = values_categorical[index][
                    start_idx : start_idx + encoder_length
                ].reshape(-1, 1)[1:]

                sample["decoder_categorical"][feature] = values_categorical[index][
                    start_idx + encoder_length : start_idx + total_sample_length
                ].reshape(-1, 1)

            target = values_target[start_idx : start_idx + encoder_length + decoder_length].reshape(-1, 1)
            sample["encoder_target"] = target[1:encoder_length]
            sample["decoder_target"] = target[encoder_length:]

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
                encoder_length=encoder_length,
                decoder_length=decoder_length,
            )
            if batch is None:
                break
            yield batch
            start_idx += 1

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        return optimizer


class RNNModel(DeepBaseModel):
    """RNN based model on LSTM cell.

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
        encoder_length: int,
        num_layers: int = 2,
        hidden_size: int = 16,
        embedding_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
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
        """Init RNN model based on LSTM cell.

        Parameters
        ----------
        input_size:
            size of the input numeric feature space: target plus extra numeric features
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        num_layers:
            number of layers
        hidden_size:
            size of the hidden state
        embedding_sizes:
            dictionary mapping categorical feature name to tuple of number of categorical classes and embedding size
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
            Pytorch lightning  trainer parameters (api reference :py:class:`lightning.pytorch.trainer.trainer.Trainer`)
        train_dataloader_params:
            parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`)
        test_dataloader_params:
            parameters for test dataloader
        val_dataloader_params:
            parameters for validation dataloader
        split_params:
            dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducible train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        """
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_sizes = embedding_sizes
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params
        super().__init__(
            net=RNNNet(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                embedding_sizes=embedding_sizes if embedding_sizes is not None else {},
                lr=lr,
                loss=nn.MSELoss() if loss is None else loss,
                optimizer_params=optimizer_params,
            ),
            decoder_length=decoder_length,
            encoder_length=encoder_length,
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

        This grid tunes parameters: ``num_layers``, ``hidden_size``, ``lr``, ``encoder_length``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "num_layers": IntDistribution(low=1, high=3),
            "hidden_size": IntDistribution(low=4, high=64, step=4),
            "lr": FloatDistribution(low=1e-5, high=1e-2, log=True),
            "encoder_length": IntDistribution(low=1, high=20),
        }
