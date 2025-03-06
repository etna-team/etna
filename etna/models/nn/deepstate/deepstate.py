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

    from etna.models.nn.deepstate import LDS
    from etna.models.nn.deepstate import CompositeSSM
    from etna.models.nn.utils import MultiEmbedding


class DeepStateBatch(TypedDict):
    """Batch specification for DeepStateModel."""

    encoder_real: "torch.Tensor"  # (batch_size, seq_length, input_size)
    decoder_real: "torch.Tensor"  # (batch_size, horizon, input_size)
    encoder_categorical: Dict[str, "torch.Tensor"]  # each (batch_size, seq_length, 1)
    decoder_categorical: Dict[str, "torch.Tensor"]  # each (batch_size, horizon, 1)
    datetime_index: "torch.Tensor"  # (batch_size, num_models, seq_length + horizon)
    encoder_target: "torch.Tensor"  # (batch_size, seq_length, 1)
    segment: "torch.Tensor"  # (batch_size)


class DeepStateNet(DeepBaseNet):
    """DeepState network."""

    def __init__(
        self,
        ssm: CompositeSSM,
        input_size: int,
        num_layers: int,
        embedding_sizes: Dict[str, Tuple[int, int]],
        n_samples: int,
        lr: float,
        optimizer_params: Optional[dict],
    ):
        """Create instance of DeepStateNet.

        Parameters
        ----------
        ssm:
            State Space Model of the system.
        input_size:
            Size of the input feature space: features for RNN part.
        num_layers:
            Number of layers in RNN.
        embedding_sizes:
            dictionary mapping categorical feature name to tuple of number of categorical classes and embedding size
        n_samples:
            Number of samples to use in predictions generation.
        lr:
            Learning rate.
        optimizer_params:
            Parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.save_hyperparameters()
        self.ssm = ssm
        self.input_size = input_size
        self.num_layers = num_layers
        self.embedding_sizes = embedding_sizes
        self.n_samples = n_samples
        self.lr = lr
        self.optimizer_params = {} if optimizer_params is None else optimizer_params
        self.latent_dim = self.ssm.latent_dim()

        self.cat_size = sum([dim for (_, dim) in self.embedding_sizes.values()])
        self.embedding: Optional[MultiEmbedding] = None
        if self.embedding_sizes:
            self.embedding = MultiEmbedding(
                embedding_sizes=self.embedding_sizes,
            )
        self.RNN = nn.LSTM(
            num_layers=self.num_layers,
            hidden_size=self.latent_dim,
            input_size=self.input_size + self.cat_size,
            batch_first=True,
        )
        self.projectors = nn.ModuleDict(
            dict(
                prior_mean=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                prior_std=nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim), nn.Softplus()
                ),
                innovation=nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim), nn.Softplus()
                ),
                noise_std=nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=1), nn.Softplus()),
                offset=nn.Linear(in_features=self.latent_dim, out_features=1),
            )
        )

    def step(self, batch: DeepStateBatch, *args, **kwargs):  # type: ignore
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
        encoder_real = batch["encoder_real"].float()  # (batch_size, seq_length, input_size)
        encoder_categorical = batch["encoder_categorical"]  # each (batch_size, seq_length, 1)
        targets = batch["encoder_target"].float()  # (batch_size, seq_length, 1)
        seq_length = targets.shape[1]
        datetime_index = batch["datetime_index"].permute(1, 0, 2)[
            :, :, :seq_length
        ]  # (num_models, batch_size, seq_length)

        encoder_embeddings = (
            self.embedding(encoder_categorical)
            if self.embedding is not None
            else torch.zeros((encoder_real.shape[0], encoder_real.shape[1], 0), device=encoder_real.device)
        )
        encoder_values = torch.concat((encoder_real, encoder_embeddings), dim=2)

        output, (_, _) = self.RNN(encoder_values)  # (batch_size, seq_length, latent_dim)
        prior_std = self.projectors["prior_std"](output[:, 0])

        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index),
            transition_coeff=self.ssm.transition_coeff(datetime_index),
            innovation_coeff=self.ssm.innovation_coeff(datetime_index) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=self.projectors["prior_mean"](output[:, 0]),
            prior_cov=torch.diag_embed(prior_std * prior_std),
            offset=self.projectors["offset"](output),
            seq_length=seq_length,
            latent_dim=self.latent_dim,
        )
        log_likelihood = lds.log_likelihood(targets=targets)
        log_likelihood = torch.mean(torch.sum(log_likelihood, dim=1))
        return -log_likelihood, targets, targets

    def forward(self, x: DeepStateBatch, *args, **kwargs):  # type: ignore
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
        encoder_real = x["encoder_real"].float()  # (batch_size, seq_length, input_size)
        encoder_categorical = x["encoder_categorical"]  # each (batch_size, seq_length, 1)
        seq_length = encoder_real.shape[1]
        targets = x["encoder_target"][:, :seq_length].float()  # (batch_size, seq_length, 1)
        decoder_real = x["decoder_real"].float()  # (batch_size, horizon, input_size)
        decoder_categorical = x["decoder_categorical"]  # each (batch_size, horizon, 1)
        datetime_index_train = x["datetime_index"].permute(1, 0, 2)[
            :, :, :seq_length
        ]  # (num_models, batch_size, seq_length)
        datetime_index_test = x["datetime_index"].permute(1, 0, 2)[
            :, :, seq_length:
        ]  # (num_models, batch_size, horizon)

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

        output, (h_n, c_n) = self.RNN(encoder_values)  # (batch_size, seq_length, latent_dim)
        prior_std = self.projectors["prior_std"](output[:, 0])
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index_train),
            transition_coeff=self.ssm.transition_coeff(datetime_index_train),
            innovation_coeff=self.ssm.innovation_coeff(datetime_index_train) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=self.projectors["prior_mean"](output[:, 0]),
            prior_cov=torch.diag_embed(prior_std * prior_std),
            offset=self.projectors["offset"](output),
            seq_length=seq_length,
            latent_dim=self.latent_dim,
        )
        _, prior_mean, prior_cov = lds.kalman_filter(targets=targets)

        output, (_, _) = self.RNN(decoder_values, (h_n, c_n))  # (batch_size, horizon, latent_dim)
        horizon = output.shape[1]
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index_test),
            transition_coeff=self.ssm.transition_coeff(datetime_index_test),
            innovation_coeff=self.ssm.innovation_coeff(datetime_index_test) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            offset=self.projectors["offset"](output),
            seq_length=horizon,
            latent_dim=self.latent_dim,
        )

        forecast = torch.mean(lds.sample(n_samples=self.n_samples), dim=0)
        return forecast

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """Make samples from segment DataFrame."""
        values_real = df.drop(
            columns=["target", "segment", "timestamp"] + list(self.embedding_sizes.keys())
        ).values.astype(np.float32)

        # Categories that were not seen during `fit` will be filled with new category
        for feature in self.embedding_sizes:
            df[feature] = df[feature].astype(np.float32).fillna(self.embedding_sizes[feature][0])

        # Columns in `values_categorical` are in the same order as in `embedding_sizes`
        values_categorical = df[self.embedding_sizes.keys()].values.T

        values_datetime = self.ssm.generate_datetime_index(df["timestamp"]).astype(int)
        values_target = df["target"].values
        segment = df["segment"].values[0]

        def _make(
            values_target: np.ndarray,
            values_real: np.ndarray,
            values_categorical: np.ndarray,
            values_datetime: np.ndarray,
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
                "segment": None,
            }
            total_length = len(values_target)
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            sample["encoder_target"] = values_target[start_idx : start_idx + encoder_length].reshape(-1, 1)
            sample["datetime_index"] = values_datetime[:, start_idx : start_idx + total_sample_length]
            sample["segment"] = segment

            sample["encoder_real"] = values_real[start_idx : start_idx + encoder_length]
            sample["decoder_real"] = values_real[start_idx + encoder_length : start_idx + total_sample_length]

            for index, feature in enumerate(self.embedding_sizes.keys()):
                sample["encoder_categorical"][feature] = values_categorical[index][
                    start_idx : start_idx + encoder_length
                ].reshape(-1, 1)

                sample["decoder_categorical"][feature] = values_categorical[index][
                    start_idx + encoder_length : start_idx + total_sample_length
                ].reshape(-1, 1)

            return sample

        start_idx = 0
        while True:
            batch = _make(
                values_target=values_target,
                values_real=values_real,
                values_categorical=values_categorical,
                values_datetime=values_datetime,
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


class DeepStateModel(DeepBaseModel):
    """DeepState model.

    Model needs label encoded inputs for categorical features, for that purposes use :py:class:`~etna.transforms.LabelEncoderTransform`.
    Feature values that weren't seen during ``fit`` should be set to NaN, to get this behaviour use encoder with *strategy="none"*.

    If there are numeric columns that are passed to ``embedding_sizes`` parameter, they will be considered only as categorical features.

    Note
    ----
    This model requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.

    Note
    ----
    This model does not currently support training on MPS.
    """

    def __init__(
        self,
        ssm: CompositeSSM,
        input_size: int,
        encoder_length: int,
        decoder_length: int,
        num_layers: int = 1,
        embedding_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        n_samples: int = 5,
        lr: float = 1e-3,
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        """Init Deep State Model.

        Parameters
        ----------
        ssm:
            state Space Model of the system
        input_size:
            size of the input feature space: features for RNN part.
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        num_layers:
            number of layers in RNN
        embedding_sizes:
            dictionary mapping categorical feature name to tuple of number of categorical classes and embedding size
        n_samples:
            number of samples to use in predictions generation
        num_layers:
            number of layers
        lr:
            learning rate
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch ligthning  trainer parameters (api reference :py:class:`lightning.pytorch.trainer.trainer.Trainer`)
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
        self.ssm = ssm
        self.input_size = input_size
        self.num_layers = num_layers
        self.embedding_sizes = embedding_sizes
        self.n_samples = n_samples
        self.lr = lr
        self.optimizer_params = optimizer_params

        # Sometimes we feed lstm an empty tensor for DeepStateModel https://github.com/pytorch/pytorch/issues/123171
        if torch.mps.is_available():
            trainer_params = {} if trainer_params is None else trainer_params
            accelerator = trainer_params.get("accelerator", None)
            if accelerator == "mps":
                raise NotImplementedError("DeepStateModel does not support MPS. Please use CPU on your MacBook.")
            elif accelerator is None:
                trainer_params["accelerator"] = "cpu"

        super().__init__(
            net=DeepStateNet(
                ssm=self.ssm,
                input_size=self.input_size,
                num_layers=self.num_layers,
                embedding_sizes=self.embedding_sizes if self.embedding_sizes is not None else {},
                n_samples=self.n_samples,
                lr=self.lr,
                optimizer_params=self.optimizer_params,
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

        This grid tunes parameters: ``lr``, ``num_layers``, ``encoder_length``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "num_layers": IntDistribution(low=1, high=3),
            "lr": FloatDistribution(low=1e-5, high=1e-2, log=True),
            "encoder_length": IntDistribution(low=1, high=20),
        }
