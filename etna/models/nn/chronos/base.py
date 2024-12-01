import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.libs.chronos import BaseChronosPipeline
from etna.libs.chronos import ChronosBoltModelForForecasting
from etna.libs.chronos import ChronosModelForForecasting
from etna.models.base import PredictionIntervalContextRequiredAbstractModel

if SETTINGS.torch_required:
    import torch


class ChronosBaseModel(PredictionIntervalContextRequiredAbstractModel):
    """Base class for Chronos-like pretrained models."""

    def __init__(
        self,
        model_name: str,
        encoder_length: int,
        device: str,
        dtype: torch.dtype,
        cache_dir: Path,
    ):
        """
        Init Chronos-like model.

        Parameters
        ----------
        model_name:
            Model name. See ``pretrained_model_name_or_path`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        encoder_length:
            Number of last timestamps to use as a context.
        device:
            Device type. See ``device_map`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        dtype:
            Torch dtype of computation. See ``torch_dtype`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        cache_dir:
            Local path to save model from huggingface during first model initialization. All following class initializations appropriate model version will be downloaded from this path.
            See ``cache_dir`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.

        Raises
        ------
        ValueError:
            If `model_name` model version is not available.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir

        self.encoder_length = encoder_length

        self.pipeline = BaseChronosPipeline.from_pretrained(
            f"amazon/{self.model_name}", device_map=self.device, torch_dtype=self.dtype, cache_dir=self.cache_dir
        )

        self.model = self.pipeline.model
        self.context: Optional[torch.Tensor] = None

    @property
    def context_size(self) -> int:
        """Context size for model."""
        return self.encoder_length

    def get_model(self) -> Union[ChronosModelForForecasting, ChronosBoltModelForForecasting]:
        """Get model."""
        return self.model

    def fit(self, ts: TSDataset):
        """Fit model.

        For this model, fit does nothing.

        Parameters
        ----------
        ts:
            Dataset with features.

        Returns
        -------
        :
            Model after fit
        """
        return self

    def predict(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
    ) -> TSDataset:
        """Make predictions using true values as autoregression context (teacher forcing).

        Parameters
        ----------
        ts:
            Dataset with features.
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components.

        Returns
        -------
        :
            Dataset with predictions.
        """
        raise NotImplementedError("Method predict isn't currently implemented!")

    def _forecast(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
        **predict_kwargs,
    ) -> TSDataset:
        """Make autoregressive forecasts.

        Parameters
        ----------
        ts:
            Dataset with features.
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components.
        **predict_kwargs:
            Predict kwargs.
        Returns
        -------
        :
            Dataset with predictions.

        Raises
        ------
        NotImplementedError:
            if return_components mode is used.
        ValueError:
            if dataset doesn't have any context timestamps.
        """
        if return_components:
            raise NotImplementedError("This mode isn't currently implemented!")

        max_context_size = len(ts.index) - prediction_size
        if max_context_size <= 0:
            raise ValueError("Dataset doesn't have any context timestamps.")

        if max_context_size < self.context_size:
            warnings.warn("Actual length of a dataset is less that context size. All history will be used as context.")
        available_context_size = min(max_context_size, self.context_size)

        target = ts.df.loc[:, pd.IndexSlice[:, "target"]]
        context = torch.tensor(target.values.T[:, :available_context_size])  # check dtype

        if prediction_interval:
            quantiles_forecast, target_forecast = self.pipeline.predict_quantiles(
                context=context,  # check dtype
                prediction_length=prediction_size,
                quantile_levels=quantiles,
                **predict_kwargs,
            )  # shape [n_segments, prediction_length, n_quantiles], [n_segments, prediction_length]
        else:
            quantiles_forecast, target_forecast = self.pipeline.predict_quantiles(
                context=context,  # check dtype
                prediction_length=prediction_size,
                **predict_kwargs,
            )  # shape [n_segments, prediction_length, n_quantiles], [n_segments, prediction_length]
        target_forecast = target_forecast.float().numpy()  # shape [n_segments, prediction_length]

        end_idx = len(ts.index)
        future_ts = ts.tsdataset_idx_slice(start_idx=end_idx - prediction_size, end_idx=end_idx)

        if prediction_interval:
            quantiles_predicts = (
                quantiles_forecast.numpy().transpose(1, 0, 2).reshape(prediction_size, -1)
            )  # shape [prediction_length, segments * n_quantiles]
            quantile_columns = [f"target_{quantile:.4g}" for quantile in quantiles]
            columns = pd.MultiIndex.from_product([ts.segments, quantile_columns], names=["segment", "feature"])
            quantiles_df = pd.DataFrame(quantiles_predicts[: len(ts.df)], columns=columns, index=future_ts.df.index)

            future_ts.add_prediction_intervals(prediction_intervals_df=quantiles_df)

        future_ts.df.loc[:, pd.IndexSlice[:, "target"]] = target_forecast.transpose(1, 0)

        return future_ts

    @staticmethod
    @abstractmethod
    def list_models() -> List[str]:
        """
        Return a list of available pretrained chronos models.

        Returns
        -------
        :
            List of available pretrained chronos models.
        """
        pass

    def save(self, path: Path):
        """Save the model. For this model, save does nothing.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        pass

    @classmethod
    def load(cls, path: Path):
        """Load the model. For this model, load does nothing.

        Parameters
        ----------
        path:
            Path to load object from.
        """
        pass

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid is empty.

        Returns
        -------
        :
            Grid to tune.
        """
        return {}
