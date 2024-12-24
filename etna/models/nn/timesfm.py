import os
import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from urllib import request

import numpy as np
import pandas as pd

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel

if SETTINGS.timesfm_required:
    from etna.libs.timesfm import TimesFmCheckpoint
    from etna.libs.timesfm import TimesFmHparams
    from etna.libs.timesfm import TimesFmTorch
    from etna.libs.timesfm.timesfm_base import freq_map

_DOWNLOAD_PATH = Path.home() / ".etna" / "timesfm"


class TimesFMModel(NonPredictionIntervalContextRequiredAbstractModel):
    """
    Class for pretrained timesfm models.

    This model is only for zero-shot forecasting: it doesn't support training on data during ``fit``.

    Official implementation: https://github.com/google-research/timesfm

    Warning
    -------
    This model doesn't support forecasting on misaligned data with `freq=None`.

    Use :py:class:`~etna.transforms.TimeSeriesImputerTransform` to fill NaNs for stable behaviour.

    Note
    ----
    This model requires ``timesfm`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        path_or_url: str,
        encoder_length: int = 512,
        device: Literal["cpu", "gpu"] = "cpu",
        batch_size: int = 128,
        static_reals: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        time_varying_reals: Optional[List[str]] = None,
        time_varying_categoricals: Optional[List[str]] = None,
        cache_dir: Path = _DOWNLOAD_PATH,
    ):
        """
        Init TimesFM model.

        Parameters
        ----------
        path_or_url:
            Path to the model. It can be huggingface repository, local path or external url.

            - If huggingface repository, the available models are:

              - 'google/timesfm-1.0-200m-pytorch'.
              During the first initialization model is downloaded from huggingface and saved to local ``cache_dir``.
              All following initializations model will be loaded from ``cache_dir``.
            - If local path, it should be a file with model weights, that can be loaded by :py:func:`torch.load`.
            - If external url, it must be a file with model weights, that can be loaded by :py:func:`torch.load`. Model will be downloaded to ``cache_dir``.
        device:
            Device type. Can be "cpu" or "gpu".
        encoder_length:
            Number of last timestamps to use as a context. It needs to be a multiplier of 32.
        batch_size:
            Batch size. It can be useful when inference is done on gpu.
        static_reals:
            Continuous features that have one unique feature value for the whole series. The first value in the series will be used for each feature.
        static_categoricals:
            Categorical features that have one unique feature value for the whole series. The first value in the series will be used for each feature.
        time_varying_reals:
            Time varying continuous features known for future.
        time_varying_categoricals:
            Time varying categorical features known for future.
        cache_dir:
            Local path to save model from huggingface during first model initialization. All following class initializations appropriate model version will be downloaded from this path.
        """
        self.path_or_url = path_or_url
        self.encoder_length = encoder_length
        self.device = device
        self.batch_size = batch_size
        self.static_reals = static_reals
        self.static_categoricals = static_categoricals
        self.time_varying_reals = time_varying_reals
        self.time_varying_categoricals = time_varying_categoricals
        self.cache_dir = cache_dir

        self._set_pipeline()

    def _set_pipeline(self):
        """Set ``tfm`` attribute."""
        if self._is_url():
            full_model_path = self._download_model_from_url()
            self.tfm = TimesFmTorch(
                hparams=TimesFmHparams(
                    context_len=self.encoder_length, per_core_batch_size=self.batch_size, backend=self.device
                ),
                checkpoint=TimesFmCheckpoint(path=full_model_path),
            )
        else:
            self.tfm = TimesFmTorch(
                hparams=TimesFmHparams(
                    context_len=self.encoder_length, per_core_batch_size=self.batch_size, backend=self.device
                ),
                checkpoint=TimesFmCheckpoint(path=self.path_or_url, local_dir=self.cache_dir),
            )

    def _is_url(self):
        """Check whether ``path_or_url`` is url."""
        return self.path_or_url.startswith("https://") or self.path_or_url.startswith("http://")

    def _download_model_from_url(self) -> str:
        """Download model from url to local cache_dir."""
        model_file = self.path_or_url.split("/")[-1]
        full_model_path = f"{self.cache_dir}/{model_file}"
        if not os.path.exists(full_model_path):
            request.urlretrieve(url=self.path_or_url, filename=full_model_path)
        return full_model_path

    @property
    def context_size(self) -> int:
        """Context size for model."""
        return self.encoder_length

    def get_model(self) -> TimesFmTorch:
        """Get model."""
        return self.tfm

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
        ValueError:
            if dataset doesn't have any context timestamps.
        NotImplementedError:
            if forecasting is done without exogenous features and dataset has None frequency.
        """
        if return_components:
            raise NotImplementedError("This mode isn't currently implemented!")

        max_context_size = len(ts.index) - prediction_size
        if max_context_size <= 0:
            raise ValueError("Dataset doesn't have any context timestamps.")

        if max_context_size < self.context_size:
            warnings.warn("Actual length of a dataset is less that context size. All history will be used as context.")

        self.tfm._set_horizon(prediction_size)

        end_idx = len(ts.index)

        static_reals_dict = (
            {column: ts.df.loc[ts.index[0], pd.IndexSlice[:, column]].values.tolist() for column in self.static_reals}
            if self.static_reals is not None
            else None
        )
        static_categoricals_dict = (
            {
                column: ts.df.loc[ts.index[0], pd.IndexSlice[:, column]].values.tolist()
                for column in self.static_categoricals
            }
            if self.static_categoricals is not None
            else None
        )
        time_varying_reals_dict = (
            {
                column: ts.df.loc[:, pd.IndexSlice[:, column]].values.swapaxes(1, 0).tolist()
                for column in self.time_varying_reals
            }
            if self.time_varying_reals is not None
            else None
        )
        time_varying_categoricals_dict = (
            {
                column: ts.df.loc[:, pd.IndexSlice[:, column]].values.swapaxes(1, 0).tolist()
                for column in self.time_varying_categoricals
            }
            if self.time_varying_categoricals is not None
            else None
        )

        future_ts = ts.tsdataset_idx_slice(start_idx=end_idx - prediction_size, end_idx=end_idx)

        if static_reals_dict or static_categoricals_dict or time_varying_reals_dict or time_varying_categoricals_dict:
            target = ts.df.loc[:, pd.IndexSlice[:, "target"]].dropna().values.swapaxes(1, 0).tolist()

            complex_forecast, _ = self.tfm.forecast_with_covariates(
                inputs=target,
                dynamic_numerical_covariates=time_varying_reals_dict,
                dynamic_categorical_covariates=time_varying_categoricals_dict,
                static_numerical_covariates=static_reals_dict,
                static_categorical_covariates=static_categoricals_dict,
                freq=[freq_map(ts.freq)] * len(ts.segments),
            )
            future_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.vstack(complex_forecast).swapaxes(1, 0)
        else:
            if ts.freq is None:
                raise NotImplementedError(
                    "Data with None frequency isn't currently implemented for forecasting without exogenous features."
                )

            target = ts.to_pandas(flatten=True, features=["target"]).dropna()
            target = target.rename(columns={"segment": "unique_id", "timestamp": "ds"})

            predictions = self.tfm.forecast_on_df(target, freq=ts.freq, value_name="target")

            predictions = predictions.rename(columns={"unique_id": "segment", "ds": "timestamp", "timesfm": "target"})
            predictions = TSDataset.to_dataset(predictions)
            future_ts.df.loc[:, pd.IndexSlice[:, "target"]] = predictions.loc[
                :, pd.IndexSlice[:, "target"]
            ].values  # .values is needed to cast predictions type of initial target type in ts
        return future_ts

    @staticmethod
    def list_models() -> List[str]:
        """
        Return a list of available pretrained timesfm models.

        Returns
        -------
        :
            List of available pretrained chronos models.
        """
        return ["google/timesfm-1.0-200m-pytorch"]

    def save(self, path: Path):
        """Save the model. This method doesn't save model's weights.

         During ``load`` weights are loaded from the path where they were saved during ``init``

        Parameters
        ----------
        path:
            Path to save object to.
        """
        self._save(path=path, skip_attributes=["tfm"])

    @classmethod
    def load(cls, path: Path):
        """Load the model.

        Parameters
        ----------
        path:
            Path to load object from.
        """
        obj: TimesFMModel = super().load(path=path)
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
