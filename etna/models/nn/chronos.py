import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import pandas as pd

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.libs.chronos.chronos import ChronosForecaster
from etna.libs.chronos.chronos import ChronosPipeline
from etna.models.base import PredictionIntervalContextRequiredAbstractModel

if SETTINGS.torch_required:
    import torch

_DOWNLOAD_PATH = Path.home() / ".etna" / "chronos-models"


class ChronosModel(PredictionIntervalContextRequiredAbstractModel):
    """
    Class for pretrained chronos models.

    This model is for zero-shot forecasting: it doesn't need calling ``fit`` method before calling ``forecast``.

    Methods ``save`` and ``load`` do nothing.
    """

    def __init__(
        self,
        model_name: str,
        encoder_length: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        limit_prediction_length: bool = False,
        batch_size: int = 128,
        cache_dir: Path = _DOWNLOAD_PATH,
    ):
        """
        Init Chronos model.

        Parameters
        ----------
        model_name:
            Model name. The available models are:

            - 'chronos-t5-tiny'
            - 'chronos-t5-mini'
            - 'chronos-t5-small'
            - 'chronos-t5-base'
            - 'chronos-t5-large'.
        device:
            Device type, "gpu" or "cpu".
        dtype:
            Torch dtype of computation.
        encoder_length:
            Number of last timestamps to use as a context.
        num_samples:
            Number of samples generated for one timestamp.
        temperature:
            Temperature of generation. Higher `temperature` will make outputs more random and diverse.
        top_k:
            Number of most likely tokens to sample from at each step of generation. Higher `top_k` will make outputs more random and diverse.
        top_p:
            The cumulative probability cutoff for token selection at each step of generation. Lower `top_p` will make outputs more random and diverse.
        limit_prediction_length:
            Whether to cancel prediction if prediction_length is greater that built-in prediction length from the model.
        batch_size:
            Batch size.
        cache_dir:
            Local path to save model from huggingface during first model initialization. All following class initializations appropriate model version will be downloaded from this path.
        Raises
        ------
        ValueError:
            If `model_name` model version is not available.
        """
        super().__init__()
        if model_name not in self.list_models():
            raise NotImplementedError(
                f"Model {model_name} is not available. To get list of available models use `list_models` method."
            )
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.encoder_length = encoder_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.limit_prediction_length = limit_prediction_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/{self.model_name}", device_map=self.device, torch_dtype=self.dtype, cache_dir=self.cache_dir
        )
        self.model = self.pipeline.model
        self.context: Optional[torch.Tensor] = None

    @property
    def context_size(self) -> int:
        """Context size for model."""
        return self.encoder_length

    def get_model(self) -> ChronosForecaster:
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

    def forecast(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
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

        Returns
        -------
        :
            Dataset with predictions.

        Raises
        ------
        NotImplementedError:
            if return_components mode is used.
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
        context = torch.tensor(target.values.T[:, :available_context_size])

        forecast = self.pipeline.predict(
            context=context,
            prediction_length=prediction_size,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            limit_prediction_length=self.limit_prediction_length,
            batch_size=self.batch_size,
        )  # shape [n_segments, num_samples, prediction_length]
        forecast = forecast.float()
        target_forecast = torch.quantile(forecast, q=0.5, dim=1).numpy()  # shape [n_segments, prediction_length]

        end_idx = len(ts.index)
        future_ts = ts.tsdataset_idx_slice(start_idx=end_idx - prediction_size, end_idx=end_idx)

        if prediction_interval:
            quantiles_predicts = torch.quantile(
                forecast, q=torch.tensor(quantiles), dim=1
            ).numpy()  # shape [n_quantiles, n_segments, prediction_length]
            quantiles_predicts = quantiles_predicts.transpose(2, 1, 0).reshape(
                prediction_size, -1
            )  # shape [prediction_length, segments * n_quantiles]
            segments = ts.segments
            quantile_columns = [f"target_{quantile:.4g}" for quantile in quantiles]
            columns = pd.MultiIndex.from_product([segments, quantile_columns], names=["segment", "feature"])
            quantiles_df = pd.DataFrame(quantiles_predicts[: len(ts.df)], columns=columns, index=future_ts.df.index)

            future_ts.add_prediction_intervals(prediction_intervals_df=quantiles_df)

        future_ts.df.loc[:, pd.IndexSlice[:, "target"]] = target_forecast.transpose(1, 0)

        return future_ts

    @staticmethod
    def list_models() -> List[str]:
        """
        Return a list of available pretrained chronos models.

        Returns
        -------
        :
            List of available pretrained chronos models.
        """
        return ["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small", "chronos-t5-base", "chronos-t5-large"]

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
