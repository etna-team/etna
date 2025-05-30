from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.models.nn.chronos.base import ChronosBaseModel

if SETTINGS.chronos_required:
    import torch

_DOWNLOAD_PATH = str(Path.home() / ".etna" / "chronos-models" / "chronos-bolt")


class ChronosBoltModel(ChronosBaseModel):
    """
    Class for pretrained chronos-bolt models.

    This model is only for zero-shot forecasting: it doesn't support training on data during ``fit``.

    Official implementation: https://github.com/amazon-science/chronos-forecasting

    Note
    ----
    This model requires ``chronos`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        path_or_url: str,
        encoder_length: int = 2048,
        device: str = "cpu",
        dtype: Optional[Union[str, torch.dtype]] = None,
        limit_prediction_length: bool = False,
        batch_size: int = 128,
        cache_dir: str = _DOWNLOAD_PATH,
    ):
        """
        Init Chronos Bolt model.

        Parameters
        ----------
        path_or_url:
            Path to the model. It can be huggingface repository, local path or external url.

            - If huggingface repository, the available models are:

              - 'amazon/chronos-bolt-tiny'
              - 'amazon/chronos-bolt-mini'
              - 'amazon/chronos-bolt-small'
              - 'amazon/chronos-bolt-base'.
              During the first initialization model is downloaded from huggingface and saved to local ``cache_dir``.
              All following initializations model will be loaded from ``cache_dir``. See ``pretrained_model_name_or_path`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
            - If local path, model will not be saved to local ``cache_dir``.
            - If external url, it must be zip archive with the same name as model directory inside. Model will be downloaded to ``cache_dir``.
        encoder_length:
            Number of last timestamps to use as a context.
        device:
            Device type. See ``device_map`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        dtype:
            Torch dtype of computation. See ``torch_dtype`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`. By default "float32" is set.
        limit_prediction_length:
            Whether to cancel prediction if prediction_length is greater that built-in prediction length from the model.
        batch_size:
            Batch size. It can be useful when inference is done on gpu.
        cache_dir:
            Local path to save model from huggingface during first model initialization. All following class initializations appropriate model version will be downloaded from this path.
            See ``cache_dir`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        """
        self.path_or_url = path_or_url
        self.encoder_length = encoder_length
        self.device = device
        self.dtype = dtype if dtype is not None else "float32"
        self.limit_prediction_length = limit_prediction_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        super().__init__(
            path_or_url=path_or_url,
            encoder_length=encoder_length,
            device=device,
            dtype=self.dtype,
            cache_dir=cache_dir,
        )

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
        """
        return self._forecast(
            ts=ts,
            prediction_size=prediction_size,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            return_components=return_components,
            limit_prediction_length=self.limit_prediction_length,
            batch_size=self.batch_size,
        )

    @staticmethod
    def list_models() -> List[str]:
        """
        Return a list of available pretrained chronos-bolt models.

        Returns
        -------
        :
            List of available pretrained chronos-bolt models.
        """
        return [
            "amazon/chronos-bolt-tiny",
            "amazon/chronos-bolt-mini",
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
        ]
