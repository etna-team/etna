from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.models.nn.chronos.base import ChronosBaseModel

if SETTINGS.chronos_required:
    import torch

_DOWNLOAD_PATH = Path.home() / ".etna" / "chronos-models" / "chronos"


class ChronosModel(ChronosBaseModel):
    """
    Class for pretrained chronos models.

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
            Device type. See ``device_map`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
        dtype:
            Torch dtype of computation. See ``torch_dtype`` parameter of :py:func:`transformers.PreTrainedModel.from_pretrained`.
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
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
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
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            limit_prediction_length=self.limit_prediction_length,
            batch_size=self.batch_size,
        )

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
