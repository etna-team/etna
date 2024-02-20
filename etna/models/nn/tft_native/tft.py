from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from etna import SETTINGS
from etna.distributions import BaseDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn

    from etna.models.base import DeepBaseModel
    from etna.models.base import DeepBaseNet
    from etna.models.nn.tft_native.layers import GateAddNorm
    from etna.models.nn.tft_native.layers import StaticCovariateEncoder
    from etna.models.nn.tft_native.layers import TemporalFusionDecoder
    from etna.models.nn.tft_native.layers import VariableSelectionNetwork
    from etna.models.nn.tft_native.loss import QuantileLoss


class TFTNativeBatch(TypedDict):
    """Batch specification for TFT."""

    decoder_target: "torch.Tensor"
    static_reals: Dict[str, "torch.Tensor"]
    static_categoricals: Dict[str, "torch.Tensor"]
    time_varying_categoricals_encoder: Dict[str, "torch.Tensor"]
    time_varying_categoricals_decoder: Dict[str, "torch.Tensor"]
    time_varying_reals_encoder: Dict[str, "torch.Tensor"]
    time_varying_reals_decoder: Dict[str, "torch.Tensor"]


class TFTNativeNet(DeepBaseNet):
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
        static_categoricals: List,
        static_reals: List,
        time_varying_categoricals_encoder: List,
        time_varying_categoricals_decoder: List,
        time_varying_reals_encoder: List,
        time_varying_reals_decoder: List,
        categorical_feature_to_id: Dict,
        loss: QuantileLoss,
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
            dropout rate in rnn layer
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
            time varying continuous features for encoder, default to `target` with lag 1
        time_varying_reals_decoder:
            time varying continuous features for decoder (known for future), default to `target` with lag 1
        categorical_feature_to_id:
            dictionary where keys are feature names and values are dictionaries mapping feature values to index from 0 to number of unique feature values
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
        self.categorical_feature_to_id = categorical_feature_to_id
        self.loss = loss
        self.optimizer_params = {} if optimizer_params is None else optimizer_params

        if self.static_categoricals:
            self.static_embeddings = nn.ModuleDict(
                {
                    feature: nn.Embedding(len(self.categorical_feature_to_id[feature]), self.hidden_size)
                    for feature in self.static_categoricals
                }
            )
        if self.static_reals:
            self.static_scalers = nn.ModuleDict(
                {feature: nn.Linear(1, self.hidden_size) for feature in self.static_reals}
            )

        if self.time_varying_categoricals_encoder:
            self.time_varying_embeddings_encoder = nn.ModuleDict(
                {
                    feature: nn.Embedding(len(self.categorical_feature_to_id[feature]), self.hidden_size)
                    for feature in self.time_varying_categoricals_encoder
                }
            )
        if self.time_varying_categoricals_decoder:
            self.time_varying_embeddings_decoder = nn.ModuleDict(
                {
                    feature: nn.Embedding(len(self.categorical_feature_to_id[feature]), self.hidden_size)
                    for feature in self.time_varying_categoricals_decoder
                }
            )

        self.time_varying_scalers_encoder = nn.ModuleDict(
            {feature: nn.Linear(1, self.hidden_size) for feature in self.time_varying_reals_encoder}
        )
        self.time_varying_scalers_decoder = nn.ModuleDict(
            {feature: nn.Linear(1, self.hidden_size) for feature in self.time_varying_reals_decoder}
        )

        if self.static_reals or self.static_categoricals:
            self.static_variable_selection = VariableSelectionNetwork(
                input_size=self.hidden_size,
                features=self.static_reals + self.static_categoricals,
                pass_context=False,
                dropout=self.dropout,
            )
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hidden_size,
            features=self.time_varying_reals_encoder + self.time_varying_categoricals_encoder,
            pass_context=True if self.static_reals + self.static_categoricals else False,
            dropout=self.dropout,
        )
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hidden_size,
            features=self.time_varying_reals_decoder + self.time_varying_categoricals_decoder,
            pass_context=True if self.static_reals + self.static_categoricals else False,
            dropout=self.dropout,
        )

        if self.static_reals or self.static_categoricals:
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
            decoder_length=self.decoder_length,
            n_heads=self.n_heads,
            pass_context=True if self.static_reals + self.static_categoricals else False,
            dropout=self.dropout,
        )

        self.gated_norm2 = GateAddNorm(input_size=self.hidden_size, output_size=self.hidden_size, dropout=0.0)

        self.output_fc = nn.Linear(self.hidden_size, 1)

    @property
    def num_timestamps(self) -> int:
        """Get number of timestamps both in encoder and decoder.

        Returns
        -------
        :
            number of timestamps.
        """
        return self.encoder_length + self.decoder_length

    def transform_features(self, x: TFTNativeBatch):
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
        if self.static_reals:
            for feature in self.static_reals:
                x["static_reals"][feature] = self.static_scalers[feature](
                    x["static_reals"][feature].float()
                )  # (batch_size, 1, hidden_size)
        if self.static_categoricals:
            for feature in self.static_categoricals:
                x["static_categoricals"][feature] = self.static_embeddings[feature](
                    x["static_categoricals"][feature].float()
                )  # (batch_size, 1, hidden_size)

        # Apply transformation to time varying data
        if self.time_varying_categoricals_encoder:
            for feature in self.time_varying_categoricals_encoder:
                x["time_varying_categoricals_encoder"][feature] = self.time_varying_embeddings_encoder[feature](
                    x["time_varying_categoricals_encoder"][feature].float()
                )  # (batch_size, encoder_length - 1, hidden_size)
        if self.time_varying_categoricals_decoder:
            for feature in self.time_varying_categoricals_decoder:
                x["time_varying_categoricals_decoder"][feature] = self.time_varying_embeddings_decoder[feature](
                    x["time_varying_categoricals_decoder"][feature].float()
                )  # (batch_size, decoder_length, hidden_size)

        for feature in self.time_varying_reals_encoder:
            x["time_varying_reals_encoder"][feature] = self.time_varying_scalers_encoder[feature](
                x["time_varying_reals_encoder"][feature].float()
            )  # (batch_size, encoder_length - 1, hidden_size)

        for feature in self.time_varying_reals_decoder:
            x["time_varying_reals_decoder"][feature] = self.time_varying_scalers_decoder[feature](
                x["time_varying_reals_decoder"][feature].float()
            )  # (batch_size, decoder_length, hidden_size)
        return x

    def forward(self, x: TFTNativeBatch, *args, **kwargs):
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
        x = self.transform_features(x)

        #  Pass static data through variable selection and covariate encoder blocks
        if self.static_reals or self.static_categoricals:
            static_features = x["static_reals"].copy()
            static_features.update(x["static_categoricals"])
            static_features = self.static_variable_selection(static_features)  # (batch_size, 1, hidden_size)

            c_s, c_c, c_h, c_e = self.static_covariate_encoder(static_features)  # (batch_size, 1, hidden_size)

        # Pass encoder data through variable selection
        encoder_features = x["time_varying_reals_encoder"].copy()
        encoder_features.update(x["time_varying_categoricals_encoder"])
        if self.static_reals or self.static_categoricals:
            encoder_features = self.encoder_variable_selection(
                x=encoder_features, context=c_s.expand(c_s.size()[0], self.encoder_length - 1, self.hidden_size)
            )  # (batch_size, encoder_length - 1, hidden_size)
        else:
            encoder_features = self.encoder_variable_selection(
                x=encoder_features
            )  # (batch_size, encoder_length - 1, hidden_size)

        residual = encoder_features

        # Pass encoder data through LSTM
        if self.static_reals or self.static_categoricals:
            c_c = c_c.permute(1, 0, 2).expand(self.num_layers, c_c.size()[0], self.hidden_size)
            c_h = c_h.permute(1, 0, 2).expand(self.num_layers, c_h.size()[0], self.hidden_size)
            encoder_features, (c_h, c_c) = self.lstm_encoder(
                encoder_features, (c_h, c_c)
            )  # (batch_size, encoder_length - 1, hidden_size)
        else:
            encoder_features, (c_h, c_c) = self.lstm_encoder(
                encoder_features
            )  # (batch_size, encoder_length - 1, hidden_size)

        # Pass encoder data through gated layer
        encoder_features = self.gated_norm1(
            x=encoder_features, residual=residual
        )  # (batch_size, encoder_length - 1, hidden_size)

        forecasts = torch.zeros((encoder_features.size()[0], self.decoder_length, 1))  # type: ignore
        for i in range(self.decoder_length):
            # Pass decoder data through variable selection
            decoder_features = x["time_varying_reals_decoder"].copy()
            decoder_features.update(x["time_varying_categoricals_decoder"])
            if self.static_reals or self.static_categoricals:
                decoder_features = self.decoder_variable_selection(
                    x=decoder_features, context=c_s.expand(c_s.size()[0], self.decoder_length, self.hidden_size)
                )  # (batch_size, decoder_length, hidden_size)
            else:
                decoder_features = self.decoder_variable_selection(
                    x=decoder_features
                )  # (batch_size, decoder_length, hidden_size)

            # Get decoder features before current decoder timestamp including it
            decoder_features = decoder_features[:, : i + 1, :]  # type: ignore
            # (batch_size, i + 1, hidden_size)

            residual = decoder_features

            # Pass decoder data through LSTM
            decoder_features, (c_h, c_c) = self.lstm_decoder(
                decoder_features, (c_h, c_c)
            )  # (batch_size, i + 1, hidden_size)

            # Pass decoder data through gated layer
            decoder_features = self.gated_norm1(x=decoder_features, residual=residual)  # (batch_size, i+1, hidden_size)

            # Pass common data through temporal fusion block
            features = torch.cat((encoder_features, decoder_features), dim=1)  # type: ignore
            residual = features  # type: ignore

            if self.static_reals or self.static_categoricals:
                features = self.temporal_fusion_decoder(
                    x=features, context=c_e.expand(features.size())
                )  # (batch_size, encoder_length + i, hidden_size)
            else:
                features = self.temporal_fusion_decoder(x=features)  # (batch_size, encoder_length + i, hidden_size)

            # Get last decoder timestamps and pass through gated layer
            decoder_features = self.gated_norm2(
                x=features[:, -1, :], residual=residual[:, -1, :]  # type: ignore
            )  # (batch_size, 1, hidden_size)

            # Get forecasts
            target_pred = self.output_fc(decoder_features)  # (batch_size, 1, 1)
            forecasts[:, i, None] = target_pred
            # Scale forecasted target and update input target for next decoder timestamp
            if i < self.decoder_length - 1:
                x["time_varying_reals_decoder"]["target"][:, i + 1, None] = self.time_varying_scalers_decoder["target"](
                    target_pred
                )

    def step(self, batch: TFTNativeBatch, *args, **kwargs):  # type: ignore
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
        target_true = batch["decoder_target"].float()  # (batch_size, decoder_length, 1)

        batch = self.transform_features(batch)

        #  Pass static data through variable selection and covariate encoder blocks
        if self.static_reals or self.static_categoricals:
            static_features = batch["static_reals"].copy()
            static_features.update(batch["static_categoricals"])
            static_features = self.static_variable_selection(static_features)  # (batch_size, 1, hidden_size)

            c_s, c_c, c_h, c_e = self.static_covariate_encoder(static_features)  # (batch_size, 1, hidden_size)

        # Pass encoder data through variable selection
        encoder_features = batch["time_varying_reals_encoder"].copy()
        encoder_features.update(batch["time_varying_categoricals_encoder"])
        if self.static_reals or self.static_categoricals:
            encoder_features = self.encoder_variable_selection(
                x=encoder_features, context=c_s.expand(c_s.size()[0], self.encoder_length - 1, self.hidden_size)
            )  # (batch_size, encoder_length - 1, hidden_size)
        else:
            encoder_features = self.encoder_variable_selection(
                x=encoder_features
            )  # (batch_size, encoder_length - 1, hidden_size)

        # Pass decoder data through variable selection
        decoder_features = batch["time_varying_reals_decoder"].copy()
        decoder_features.update(batch["time_varying_categoricals_decoder"])
        if self.static_reals or self.static_categoricals:
            decoder_features = self.decoder_variable_selection(
                x=decoder_features, context=c_s.expand(c_s.size()[0], self.decoder_length, self.hidden_size)
            )  # (batch_size, decoder_length, hidden_size)
        else:
            decoder_features = self.decoder_variable_selection(
                x=decoder_features
            )  # (batch_size, decoder_length, hidden_size)

        residual = torch.cat((encoder_features, decoder_features), dim=1)  # type: ignore

        # Pass encoder and decoder data through LSTM
        if self.static_reals or self.static_categoricals:
            c_c = c_c.permute(1, 0, 2).expand(self.num_layers, c_c.size()[0], self.hidden_size)
            c_h = c_h.permute(1, 0, 2).expand(self.num_layers, c_h.size()[0], self.hidden_size)
            encoder_features, (c_h, c_c) = self.lstm_encoder(
                encoder_features, (c_h, c_c)
            )  # (batch_size, encoder_length - 1, hidden_size)
        else:
            encoder_features, (c_h, c_c) = self.lstm_encoder(
                encoder_features
            )  # (batch_size, encoder_length - 1, hidden_size)
        decoder_features, (_, _) = self.lstm_decoder(
            decoder_features, (c_h, c_c)
        )  # (batch_size, decoder_length, hidden_size)

        # Pass common data through gated layer
        features = torch.cat((encoder_features, decoder_features), dim=1)  # type: ignore
        features = self.gated_norm1(x=features, residual=residual)  # (batch_size, num_timestamps, hidden_size)

        residual = features

        # Pass common data through temporal fusion block
        if self.static_reals or self.static_categoricals:
            features = self.temporal_fusion_decoder(
                x=features, context=c_e.expand(features.size())
            )  # (batch_size, num_timestamps, hidden_size)
        else:
            features = self.temporal_fusion_decoder(x=features)  # (batch_size, num_timestamps, hidden_size)

        # Get decoder timestamps and pass through gated layer
        decoder_features = self.gated_norm2(
            x=features[:, -self.decoder_length, :], residual=residual[:, -self.decoder_length, :]
        )  # (batch_size, decoder_length, hidden_size)

        target_pred = self.output_fc(decoder_features)  # (batch_size, decoder_length, 1)

        loss = self.loss(inputs=target_true, pred=target_pred)
        return loss, target_true, target_pred

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """Make samples from segment DataFrame."""
        values_target = df["target"].values
        df["target"] = df["target"].shift(1)

        for feature in self.categorical_feature_to_id:
            df[feature] = df[feature].map(self.categorical_feature_to_id[feature])

        def _make(
            values_target: np.ndarray,
            df: pd.DataFrame,
            start_idx: int,
            encoder_length: int,
            decoder_length: int,
        ) -> Optional[dict]:

            sample: Dict[str, Any] = {
                "decoder_target": list(),
                "static_reals": dict(),
                "static_categoricals": dict(),
                "time_varying_categoricals_encoder": dict(),
                "time_varying_categoricals_decoder": dict(),
                "time_varying_reals_encoder": dict(),
                "time_varying_reals_decoder": dict(),
            }
            total_length = len(values_target)
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            sample["decoder_target"] = values_target[encoder_length:].reshape(-1, 1)  # (decoder_length, 1)

            for feature in self.static_reals:
                sample["static_reals"][feature] = df[[feature]].values[:1]  # (1, 1)

            for feature in self.static_categoricals:
                sample["static_categoricals"][feature] = df[[feature]].values[:1]  # (1, 1)

            for feature in self.time_varying_categoricals_encoder:
                sample["time_varying_categoricals_encoder"][feature] = df[[feature]].values[
                    1:encoder_length
                ]  # (encoder_length-1, 1)

            for feature in self.time_varying_categoricals_decoder:
                sample["time_varying_categoricals_decoder"][feature] = df[[feature]].values[
                    encoder_length:
                ]  # (decoder_length, 1)

            for feature in self.time_varying_reals_encoder:
                sample["time_varying_reals_encoder"][feature] = df[[feature]].values[
                    1:encoder_length
                ]  # (encoder_length-1, 1)

            for feature in self.time_varying_reals_decoder:
                sample["time_varying_reals_decoder"][feature] = df[[feature]].values[
                    encoder_length:
                ]  # (decoder_length, 1)

            return sample

        start_idx = 0
        while True:
            batch = _make(
                values_target=values_target,
                df=df,
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


class TFTNativeModel(DeepBaseModel):
    """TFT model.

    Note
    ----
    This model requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
        hidden_size: int = 16,
        lr: float = 1e-3,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        time_varying_reals_encoder: List[str] = ["target"],
        time_varying_reals_decoder: List[str] = ["target"],
        categorical_feature_to_id: Dict[str, Dict[str, int]] = {},
        loss: Optional["QuantileLoss"] = None,
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
            categorical features for the whole series, e.g. `segment`
        static_reals:
            continuous features for the whole series
        time_varying_categoricals_encoder:
            time varying categorical features for encoder
        time_varying_categoricals_decoder:
            time varying categorical features for decoder (known for future)
        time_varying_reals_encoder:
            time varying continuous features for encoder, default to `target` with lag 1
        time_varying_reals_decoder:
            time varying continuous features for decoder (known for future), default to `target` with lag 1
        categorical_feature_to_id:
            dictionary where keys are feature names and values are dictionaries mapping feature values to index from 0 to number of unique feature values
        loss:
            loss function
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch lightning trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`)
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
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.lr = lr
        self.optimizer_params = optimizer_params
        self.loss = loss
        super().__init__(
            net=TFTNativeNet(
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
                hidden_size=hidden_size,
                lr=lr,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_categoricals_encoder=time_varying_categoricals_encoder,
                time_varying_categoricals_decoder=time_varying_categoricals_decoder,
                time_varying_reals_encoder=time_varying_reals_encoder,
                time_varying_reals_decoder=time_varying_reals_decoder,
                categorical_feature_to_id=categorical_feature_to_id,
                optimizer_params=optimizer_params,
                loss=QuantileLoss() if loss is None else loss,
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

        This grid tunes parameters: ``num_layers``, ``hidden_size``, ``lr``, ``encoder_length``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "num_layers": IntDistribution(low=1, high=3),
            "n_heads": IntDistribution(low=1, high=3),
            "hidden_size": IntDistribution(low=4, high=64, step=4),
            "lr": FloatDistribution(low=1e-5, high=1e-2, log=True),
            "encoder_length": IntDistribution(low=1, high=20),
            "dropout": FloatDistribution(low=0.0, high=0.3),
        }
