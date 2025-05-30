import os
import warnings
import zipfile
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union
from urllib import request

import pandas as pd

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.models.base import PredictionIntervalContextRequiredAbstractModel

if SETTINGS.chronos_required:
    import torch

    from etna.libs.chronos import BaseChronosPipeline
    from etna.libs.chronos import ChronosBoltModelForForecasting
    from etna.libs.chronos import ChronosModelForForecasting


class ChronosBaseModel(PredictionIntervalContextRequiredAbstractModel):
    """Base class for Chronos-like pretrained models."""

    def __init__(
        self,
        path_or_url: str,
        encoder_length: int,
        device: str,
        dtype: Union[str, torch.dtype],
        cache_dir: str,
    ):
        """
        Init Chronos-like model.

        Parameters
        ----------
        path_or_url:
            Path to the model. It can be huggingface repository, local path or external url.

            - If huggingface repository, see available models in ``list_models`` of appropriate model class.
            During the first initialization model is downloaded from huggingface and saved to local ``cache_dir``.
            All following initializations model will be loaded from ``cache_dir``. See ``pretrained_model_name_or_path`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
            - If local path, model will not be saved to local ``cache_dir``.
            - If external url, it must be zip archive with the same name as model directory inside. Model will be downloaded to ``cache_dir``.
        encoder_length:
            Number of last timestamps to use as a context.
        device:
            Device type. See ``device_map`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        dtype:
            Torch dtype of computation. See ``torch_dtype`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        cache_dir:
            Local path to save model from huggingface during first model initialization. All following class initializations appropriate model version will be downloaded from this path.
            See ``cache_dir`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        """
        super().__init__()
        self.path_or_url = path_or_url
        self.encoder_length = encoder_length
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir

        self._set_pipeline()

    def _set_pipeline(self):
        """Set ``pipeline`` attribute."""
        if self._is_url():
            full_model_path = self._download_model_from_url()
            self.pipeline = BaseChronosPipeline.from_pretrained(
                full_model_path, device_map=self.device, torch_dtype=self.dtype, cache_dir=self.cache_dir
            )
        else:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.path_or_url, device_map=self.device, torch_dtype=self.dtype, cache_dir=self.cache_dir
            )

    def _is_url(self):
        """Check whether ``path_or_url`` is url."""
        return self.path_or_url.startswith("https://") or self.path_or_url.startswith("http://")

    def _download_model_from_url(self) -> str:
        """Download model from url to local cache_dir."""
        model_file = self.path_or_url.split("/")[-1]
        model_dir = model_file.split(".zip")[0]
        full_model_path = f"{self.cache_dir}/{model_dir}"
        if not os.path.exists(full_model_path):
            try:
                request.urlretrieve(url=self.path_or_url, filename=model_file)

                with zipfile.ZipFile(model_file, "r") as zip_ref:
                    zip_ref.extractall(self.cache_dir)
            finally:
                os.remove(model_file)
        return full_model_path

    @property
    def context_size(self) -> int:
        """Context size for model."""
        return self.encoder_length

    def get_model(self) -> Union[ChronosModelForForecasting, ChronosBoltModelForForecasting]:
        """Get model."""
        return self.pipeline.model

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
            Additional predict parameters for Chronos and Chronos-Bolt models.
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

        max_context_size = len(ts.timestamps) - prediction_size
        if max_context_size <= 0:
            raise ValueError("Dataset doesn't have any context timestamps.")

        if max_context_size < self.context_size:
            warnings.warn("Actual length of a dataset is less that context size. All history will be used as context.")

        target = ts._df.loc[:, pd.IndexSlice[:, "target"]].dropna()
        context = torch.tensor(target.values.T)

        if prediction_interval:
            quantiles_forecast, target_forecast = self.pipeline.predict_quantiles(
                context=context,
                prediction_length=prediction_size,
                quantile_levels=quantiles,
                **predict_kwargs,
            )  # shape [n_segments, prediction_length, n_quantiles], [n_segments, prediction_length]
        else:
            quantiles_forecast, target_forecast = self.pipeline.predict_quantiles(
                context=context,
                prediction_length=prediction_size,
                **predict_kwargs,
            )  # shape [n_segments, prediction_length, n_quantiles], [n_segments, prediction_length]

        end_idx = len(ts.timestamps)
        future_ts = ts.tsdataset_idx_slice(start_idx=end_idx - prediction_size, end_idx=end_idx)

        if prediction_interval:
            quantiles_predicts = (
                quantiles_forecast.numpy().transpose(1, 0, 2).reshape(prediction_size, -1)
            )  # shape [prediction_length, segments * n_quantiles]
            quantile_columns = [f"target_{quantile:.4g}" for quantile in quantiles]
            columns = pd.MultiIndex.from_product([ts.segments, quantile_columns], names=["segment", "feature"])
            quantiles_df = pd.DataFrame(quantiles_predicts[: ts.size()[0]], columns=columns, index=future_ts.timestamps)

            future_ts.add_prediction_intervals(prediction_intervals_df=quantiles_df)

        future_ts._df.loc[:, pd.IndexSlice[:, "target"]] = target_forecast.numpy().transpose(1, 0)

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
        """Save the model. This method doesn't save model's weights.

         During ``load`` weights are loaded from the path where they were saved during ``init``

        Parameters
        ----------
        path:
            Path to save object to.
        """
        self._save(path=path, skip_attributes=["pipeline"])

    @classmethod
    def load(cls, path: Path):
        """Load the model.

        Parameters
        ----------
        path:
            Path to load object from.
        """
        obj: ChronosBaseModel = super().load(path=path)
        obj._set_pipeline()
        return obj

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid is empty.

        Returns
        -------
        :
            Grid to tune.
        """
        return {}
