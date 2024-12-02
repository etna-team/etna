from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.models.nn.chronos.base import ChronosBaseModel

if SETTINGS.chronos_required:
    import torch

_DOWNLOAD_PATH = Path.home() / ".etna" / "chronos-models" / "chronos-bolt"


class ChronosBoltModel(ChronosBaseModel):
    """
    Class for pretrained chronos-bolt models.

    This model is only for zero-shot forecasting: it doesn't support training on data during ``fit``.

    Methods ``save`` and ``load`` do nothing.

    Official implementation: https://github.com/amazon-science/chronos-forecasting

    Note
    ----
    This model requires ``chronos`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        model_name: str,
        encoder_length: int = 2048,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        limit_prediction_length: bool = False,
        batch_size: int = 128,
        cache_dir: Path = _DOWNLOAD_PATH,
    ):
        """
        Init Chronos model.

        Parameters
        ----------
        model_name:
            Model name.
        encoder_length:
            Number of last timestamps to use as a context.
        device:
            Device type. See ``device_map`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        dtype:
            Torch dtype of computation. See ``torch_dtype`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        limit_prediction_length:
            Whether to cancel prediction if prediction_length is greater that built-in prediction length from the model.
        batch_size:
            Batch size. It can be useful when inference is done on gpu.
        cache_dir:
            Local path to save model from huggingface during first model initialization. All following class initializations appropriate model version will be downloaded from this path.
            See ``cache_dir`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.

        Raises
        ------
        ValueError:
            If `model_name` model version is not available.
        """
        if model_name not in self.list_models():
            raise NotImplementedError(
                f"Model {model_name} is not available. To get list of available models use `list_models` method."
            )
        self.model_name = model_name
        self.encoder_length = encoder_length
        self.device = device
        self.dtype = dtype
        self.limit_prediction_length = limit_prediction_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self.context: Optional[torch.Tensor] = None

        super().__init__(
            model_name=model_name, encoder_length=encoder_length, device=device, dtype=dtype, cache_dir=cache_dir
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
        return ["chronos-bolt-tiny", "chronos-bolt-mini", "chronos-bolt-small", "chronos-bolt-base"]
