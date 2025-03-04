from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from etna import SETTINGS
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn

    from etna.models.base import DeepBaseModel
    from etna.models.base import DeepBaseNet
    from etna.models.nn.tft.layers import GateAddNorm
    from etna.models.nn.tft.layers import StaticCovariateEncoder
    from etna.models.nn.tft.layers import TemporalFusionDecoder
    from etna.models.nn.tft.layers import VariableSelectionNetwork


class TFTBatch(TypedDict):
    """Batch specification for TFT."""

    segment: str
    decoder_target: "torch.Tensor"
    static_reals: Dict[str, "torch.Tensor"]
    static_categoricals: Dict[str, "torch.Tensor"]
    time_varying_categoricals_encoder: Dict[str, "torch.Tensor"]
    time_varying_categoricals_decoder: Dict[str, "torch.Tensor"]
    time_varying_reals_encoder: Dict[str, "torch.Tensor"]
    time_varying_reals_decoder: Dict[str, "torch.Tensor"]


class TFTNet(DeepBaseNet):
    """TFT based Lightning module."""

    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        hidden_size: int,
        lr: float,
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        num_embeddings: Dict[str, int],
        loss: nn.Module,
        optimizer_params: Optional[dict],
    ) -> None:
        """Init TFT.

        Parameters
        ----------
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        n_heads:
            number of heads in Multi-Head Attention
        num_layers:
            number of layers in LSTM layer
        dropout:
            dropout rate
        hidden_size:
            size of the hidden state
        lr:
            learning rate
        static_categoricals:
            categorical features for the whole series, e.g. `segment`
        static_reals:
            continuous features for the whole series
        time_varying_categoricals_encoder:
            time varying categorical features for encoder
        time_varying_categoricals_decoder:
            time varying categorical features for decoder (known for future)
        time_varying_reals_encoder:
            time varying continuous features for encoder, default to `target`
        time_varying_reals_decoder:
            time varying continuous features for decoder (known for future)
        num_embeddings:
            dictionary where keys are feature names and values are number of unique values of that features
        loss:
            loss function
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.lr = lr
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_categoricals_encoder = time_varying_categoricals_encoder
        self.time_varying_categoricals_decoder = time_varying_categoricals_decoder
        self.time_varying_reals_encoder = time_varying_reals_encoder
        self.time_varying_reals_decoder = time_varying_reals_decoder
        self.num_embeddings = num_embeddings
        self.loss = loss
        self.optimizer_params = {} if optimizer_params is None else optimizer_params

        self.static_scalers = nn.ModuleDict({feature: nn.Linear(1, self.hidden_size) for feature in self.static_reals})
        self.static_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(self.num_embeddings[feature] + 1, self.hidden_size)
                for feature in self.static_categoricals
            }
        )

        self.time_varying_scalers_encoder = nn.ModuleDict(
            {feature: nn.Linear(1, self.hidden_size) for feature in self.time_varying_reals_encoder}
        )
        self.time_varying_embeddings_encoder = nn.ModuleDict(
            {
                feature: nn.Embedding(self.num_embeddings[feature] + 1, self.hidden_size)
                for feature in self.time_varying_categoricals_encoder
            }
        )

        self.time_varying_scalers_decoder = nn.ModuleDict(
            {feature: nn.Linear(1, self.hidden_size) for feature in self.time_varying_reals_decoder}
        )
        self.time_varying_embeddings_decoder = nn.ModuleDict(
            {
                feature: nn.Embedding(self.num_embeddings[feature] + 1, self.hidden_size)
                for feature in self.time_varying_categoricals_decoder
            }
        )

        self.static_variable_selection: Optional[VariableSelectionNetwork] = None
        if self._num_static > 0:
            self.static_variable_selection = VariableSelectionNetwork(
                input_size=self.hidden_size,
                features=static_reals + static_categoricals,
                dropout=self.dropout,
            )
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hidden_size,
            features=self.time_varying_reals_encoder + self.time_varying_categoricals_encoder,
            context_size=self.hidden_size if self._num_static > 0 else None,
            dropout=self.dropout,
        )

        self.decoder_variable_selection: Optional[VariableSelectionNetwork] = None
        if self._num_decoder_features > 0:
            self.decoder_variable_selection = VariableSelectionNetwork(
                input_size=self.hidden_size,
                features=self.time_varying_reals_decoder + self.time_varying_categoricals_decoder,
                context_size=self.hidden_size if self._num_static > 0 else None,
                dropout=self.dropout,
            )

        self.static_covariate_encoder: Optional[StaticCovariateEncoder] = None
        if self._num_static > 0:
            self.static_covariate_encoder = StaticCovariateEncoder(
                input_size=hidden_size,
                dropout=self.dropout,
            )

        self.lstm_encoder = nn.LSTM(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            input_size=self.hidden_size,
            batch_first=True,
            dropout=self.dropout,
        )
        self.lstm_decoder = nn.LSTM(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            input_size=self.hidden_size,
            batch_first=True,
            dropout=self.dropout,
        )

        self.gated_norm1 = GateAddNorm(input_size=self.hidden_size, output_size=self.hidden_size, dropout=self.dropout)

        self.temporal_fusion_decoder = TemporalFusionDecoder(
            input_size=self.hidden_size,
            n_heads=self.n_heads,
            context_size=self.hidden_size if self._num_static > 0 else None,
            dropout=self.dropout,
        )

        self.gated_norm2 = GateAddNorm(input_size=self.hidden_size, output_size=self.hidden_size, dropout=0.0)

        self.output_fc = nn.Linear(self.hidden_size, 1)

    @property
    def _num_static(self) -> int:
        """Get number of static features.

        Returns
        -------
        :
            number of static features.
        """
        return len(self.static_reals + self.static_categoricals)

    @property
    def _num_decoder_features(self) -> int:
        """Get number of features in decoder.

        Returns
        -------
        :
            number of features in decoder.
        """
        return len(self.time_varying_reals_decoder + self.time_varying_categoricals_decoder)

    def _transform_features(self, x: TFTBatch) -> TFTBatch:
        """Apply embedding layer to categorical input features and linear transformation to continuous features.

        Parameters
        ----------
        x:
            batch of data

        Returns
        -------
        :
            transformed batch of data
        """
        # Apply transformation to static data
        for feature in self.static_reals:
            x["static_reals"][feature] = self.static_scalers[feature](
                x["static_reals"][feature].float()
            )  # (batch_size, 1, hidden_size)
        for feature in self.static_categoricals:
            x["static_categoricals"][feature] = self.static_embeddings[feature](
                x["static_categoricals"][feature].int().squeeze(2)
            )  # (batch_size, 1, hidden_size)

        # Apply transformation to time varying data
        for feature in self.time_varying_categoricals_encoder:
            x["time_varying_categoricals_encoder"][feature] = self.time_varying_embeddings_encoder[feature](
                x["time_varying_categoricals_encoder"][feature].int().squeeze(2)
            )  # (batch_size, encoder_length, hidden_size)
        for feature in self.time_varying_categoricals_decoder:
            x["time_varying_categoricals_decoder"][feature] = self.time_varying_embeddings_decoder[feature](
                x["time_varying_categoricals_decoder"][feature].int().squeeze(2)
            )  # (batch_size, decoder_length, hidden_size)

        for feature in self.time_varying_reals_encoder:
            x["time_varying_reals_encoder"][feature] = self.time_varying_scalers_encoder[feature](
                x["time_varying_reals_encoder"][feature].float()
            )  # (batch_size, encoder_length, hidden_size)
        for feature in self.time_varying_reals_decoder:
            x["time_varying_reals_decoder"][feature] = self.time_varying_scalers_decoder[feature](
                x["time_varying_reals_decoder"][feature].float()
            )  # (batch_size, decoder_length, hidden_size)
        return x

    def forward(self, x: TFTBatch, *args, **kwargs) -> torch.Tensor:
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
        target_true = x["decoder_target"].float()  # (batch_size, decoder_length, 1)
        decoder_length = target_true.size()[1]
        batch_size = target_true.size()[0]
        batch = self._transform_features(x)

        #  Pass static data through variable selection and covariate encoder blocks
        if self.static_variable_selection is not None and self.static_covariate_encoder is not None:
            static_features = batch["static_reals"].copy()
            static_features.update(batch["static_categoricals"])
            static_features = self.static_variable_selection(static_features)  # (batch_size, 1, hidden_size)

            c_s, c_c, c_h, c_e = self.static_covariate_encoder(static_features)  # (batch_size, 1, hidden_size)

        # Pass encoder data through variable selection
        encoder_features = batch["time_varying_reals_encoder"].copy()
        encoder_features.update(batch["time_varying_categoricals_encoder"])
        if self._num_static > 0:
            encoder_output = self.encoder_variable_selection(
                x=encoder_features, context=c_s.expand(batch_size, self.encoder_length, self.hidden_size)
            )  # (batch_size, encoder_length, hidden_size)
        else:
            encoder_output = self.encoder_variable_selection(
                x=encoder_features
            )  # (batch_size, encoder_length, hidden_size)
        if self.decoder_variable_selection is not None:
            # Pass decoder data through variable selection
            decoder_features = batch["time_varying_reals_decoder"].copy()
            decoder_features.update(batch["time_varying_categoricals_decoder"])
            if self._num_static > 0:
                decoder_output = self.decoder_variable_selection(
                    x=decoder_features, context=c_s.expand(batch_size, decoder_length, self.hidden_size)
                )  # (batch_size, decoder_length, hidden_size)
            else:
                decoder_output = self.decoder_variable_selection(
                    x=decoder_features
                )  # (batch_size, decoder_length, hidden_size)
        else:
            decoder_output = torch.zeros(batch_size, decoder_length, self.hidden_size, device=encoder_output.device)
        residual = torch.cat((encoder_output, decoder_output), dim=1)

        # Pass encoder and decoder data through LSTM
        if self._num_static > 0:
            c_c = c_c.permute(1, 0, 2).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
            c_h = c_h.permute(1, 0, 2).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
            encoder_output, (c_h, c_c) = self.lstm_encoder(
                encoder_output, (c_h, c_c)
            )  # (batch_size, encoder_length, hidden_size)
        else:
            encoder_output, (c_h, c_c) = self.lstm_encoder(encoder_output)  # (batch_size, encoder_length, hidden_size)
        decoder_output, (_, _) = self.lstm_decoder(
            decoder_output, (c_h, c_c)
        )  # (batch_size, decoder_length, hidden_size)

        # Pass common data through gated layer
        features = torch.cat((encoder_output, decoder_output), dim=1)
        features = self.gated_norm1(x=features, residual=residual)  # (batch_size, num_timestamps, hidden_size)

        residual = features

        # Pass common data through temporal fusion block
        if self._num_static > 0:
            features = self.temporal_fusion_decoder(
                x=features, context=c_e.expand(features.size())
            )  # (batch_size, num_timestamps, hidden_size)
        else:
            features = self.temporal_fusion_decoder(x=features)  # (batch_size, num_timestamps, hidden_size)

        # Get decoder timestamps and pass through gated layer
        decoder_features = self.gated_norm2(
            x=features[:, -decoder_length:, :], residual=residual[:, -decoder_length:, :]
        )  # (batch_size, decoder_length, hidden_size)

        target_pred = self.output_fc(decoder_features)  # (batch_size, decoder_length, 1)

        return target_pred

    def step(self, batch: TFTBatch, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
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
        target_pred = self.forward(batch)  # (batch_size, decoder_length, 1)
        target_true = batch["decoder_target"].float()  # (batch_size, decoder_length, 1)
        loss = self.loss(target_pred, target_true)
        return loss, target_true, target_pred

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """Make samples from segment DataFrame."""
        segment = df["segment"].values[0]
        for feature in self.num_embeddings:
            df[feature] = df[feature].astype(np.float32).fillna(self.num_embeddings[feature])

        reals_columns = list(set(self.static_reals + self.time_varying_reals_encoder + self.time_varying_reals_decoder))
        categ_columns = list(
            set(
                self.static_categoricals
                + self.time_varying_categoricals_encoder
                + self.time_varying_categoricals_decoder
            )
        )

        df = df[reals_columns + categ_columns]
        column_to_index = {column: index for index, column in enumerate(df.columns)}
        values = df.values.T.astype(np.float32)

        def _make(
            values: np.ndarray,
            column_to_index: dict,
            segment: str,
            start_idx: int,
            encoder_length: int,
            decoder_length: int,
        ) -> Optional[dict]:

            sample: Dict[str, Any] = {
                "segment": None,
                "decoder_target": list(),
                "static_reals": dict(),
                "static_categoricals": dict(),
                "time_varying_categoricals_encoder": dict(),
                "time_varying_categoricals_decoder": dict(),
                "time_varying_reals_encoder": dict(),
                "time_varying_reals_decoder": dict(),
            }
            total_length = values.shape[1]
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            sample["segment"] = segment
            sample["decoder_target"] = values[column_to_index["target"]][
                start_idx + encoder_length : start_idx + total_sample_length
            ].reshape(
                -1, 1
            )  # (decoder_length, 1)

            for feature in self.static_reals:
                sample["static_reals"][feature] = values[column_to_index[feature]][:1].reshape(-1, 1)  # (1, 1)

            for feature in self.static_categoricals:
                sample["static_categoricals"][feature] = values[column_to_index[feature]][:1].reshape(-1, 1)  # (1, 1)

            for feature in self.time_varying_categoricals_encoder:
                sample["time_varying_categoricals_encoder"][feature] = values[column_to_index[feature]][
                    start_idx : start_idx + encoder_length
                ].reshape(
                    -1, 1
                )  # (encoder_length, 1)

            for feature in self.time_varying_categoricals_decoder:
                sample["time_varying_categoricals_decoder"][feature] = values[column_to_index[feature]][
                    start_idx + encoder_length : start_idx + total_sample_length
                ].reshape(
                    -1, 1
                )  # (decoder_length, 1)

            for feature in self.time_varying_reals_encoder:
                sample["time_varying_reals_encoder"][feature] = values[column_to_index[feature]][
                    start_idx : start_idx + encoder_length
                ].reshape(
                    -1, 1
                )  # (encoder_length, 1)

            for feature in self.time_varying_reals_decoder:
                sample["time_varying_reals_decoder"][feature] = values[column_to_index[feature]][
                    start_idx + encoder_length : start_idx + total_sample_length
                ].reshape(
                    -1, 1
                )  # (decoder_length, 1)

            return sample

        start_idx = 0
        while True:
            batch = _make(
                values=values,
                column_to_index=column_to_index,
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


class TFTModel(DeepBaseModel):
    """TFT model. For more details read the `paper <https://arxiv.org/abs/1912.09363>`_.

    Model needs label encoded inputs for categorical features, for that purposes use :py:class:`~etna.transforms.LabelEncoderTransform`.
    Feature values that were not seen during `fit` should be set to NaN for expected behaviour with `strategy="none"`

    Passed feature values aren't validated on being static or being correctly label encoded.

    Note
    ----
    This model requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.

    Note
    ----
    This model was previously named ``TFTNativeModel``. The original ``TFTModel`` based on ``pytorch_forecasting`` was removed.

    Note
    ----
    This model does not currently support training on MPS.
    """

    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        hidden_size: int = 160,
        lr: float = 1e-3,
        static_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        time_varying_categoricals_encoder: Optional[List[str]] = None,
        time_varying_categoricals_decoder: Optional[List[str]] = None,
        time_varying_reals_encoder: Optional[List[str]] = None,
        time_varying_reals_decoder: Optional[List[str]] = None,
        num_embeddings: Optional[Dict[str, int]] = None,
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
        """Init TFT model.

        Parameters
        ----------
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        n_heads:
            number of heads in Multi-Head Attention
        num_layers:
            number of layers in LSTM layer
        dropout:
            dropout rate
        hidden_size:
            size of the hidden state
        lr:
            learning rate
        static_categoricals:
            categorical features that have one unique feature value for the whole series, e.g. `segment`.
            The first value in the series is passed to batch for each feature.
        static_reals:
            continuous features that have one unique feature value for the whole series.
            The first value in the series is passed to batch for each feature.
        time_varying_categoricals_encoder:
            time varying categorical features for encoder
        time_varying_categoricals_decoder:
            time varying categorical features for decoder (known for future)
        time_varying_reals_encoder:
            time varying continuous features for encoder, default to `target`
        time_varying_reals_decoder:
            time varying continuous features for decoder (known for future)
        num_embeddings:
            dictionary where keys are feature names and values are number of unique values of that features
        loss:
            loss function
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch lightning trainer parameters (api reference :py:class:`lightning.pytorch.trainer.trainer.Trainer`)
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
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.lr = lr
        self.static_categoricals = static_categoricals if static_categoricals is not None else []
        self.static_reals = static_reals if static_reals is not None else []
        self.time_varying_categoricals_encoder = (
            time_varying_categoricals_encoder if time_varying_categoricals_encoder is not None else []
        )
        self.time_varying_categoricals_decoder = (
            time_varying_categoricals_decoder if time_varying_categoricals_decoder is not None else []
        )
        self.time_varying_reals_encoder = (
            time_varying_reals_encoder if time_varying_reals_encoder is not None else ["target"]
        )
        self.time_varying_reals_decoder = time_varying_reals_decoder if time_varying_reals_decoder is not None else []
        self.num_embeddings = num_embeddings if num_embeddings is not None else {}
        self.optimizer_params = optimizer_params
        self.loss = nn.MSELoss() if loss is None else loss

        if torch.mps.is_available():
            trainer_params = {} if trainer_params is None else trainer_params
            accelerator = trainer_params.get("accelerator", None)
            if accelerator == "mps":
                raise NotImplementedError("TFTModel does not support MPS. Please use CPU on your MacBook.")
            elif accelerator is None:
                trainer_params["accelerator"] = "cpu"

        super().__init__(
            net=TFTNet(
                encoder_length=self.encoder_length,
                decoder_length=self.decoder_length,
                n_heads=self.n_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                hidden_size=self.hidden_size,
                lr=self.lr,
                static_categoricals=self.static_categoricals,
                static_reals=self.static_reals,
                time_varying_categoricals_encoder=self.time_varying_categoricals_encoder,
                time_varying_categoricals_decoder=self.time_varying_categoricals_decoder,
                time_varying_reals_encoder=self.time_varying_reals_encoder,
                time_varying_reals_decoder=self.time_varying_reals_decoder,
                num_embeddings=self.num_embeddings,
                optimizer_params=self.optimizer_params,
                loss=self.loss,
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

        This grid tunes parameters: ``num_layers``, ``n_heads``, ``hidden_size``, ``lr``, ``dropout``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "num_layers": IntDistribution(low=1, high=3),
            "n_heads": CategoricalDistribution([1, 4]),
            "hidden_size": CategoricalDistribution([16, 20, 40, 80, 160, 240, 320]),
            "lr": FloatDistribution(low=1e-4, high=1e-2, log=True),
            "dropout": FloatDistribution(low=0.1, high=0.9, step=0.1),
        }
