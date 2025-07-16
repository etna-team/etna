import hashlib
import os
import warnings
from typing import Dict
from typing import List
from typing import Optional
from urllib import request

import numpy as np
from sklearn.base import ClassifierMixin

from etna.datasets import TSDataset
from etna.experimental.classification.classification import TimeSeriesBinaryClassifier
from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor
from etna.experimental.classification.utils import crop_nans_single_series

# Known model hashes for integrity verification
# To add a hash for a model URL, download the file and compute its MD5 hash
_KNOWN_MODEL_HASHES = {
    # Add known model URL -> hash mappings here
    # Example: "http://example.com/model.pickle": "abcd1234...",
}


def _verify_file_hash(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """
    Verify file integrity using MD5 hash.

    Parameters
    ----------
    file_path:
        Path to the file to verify
    expected_hash:
        Expected MD5 hash. If None, verification is skipped.

    Returns
    -------
    :
        True if hash matches or no expected hash provided, False otherwise
    """
    if expected_hash is None:
        return True

    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception:
        return False


class PredictabilityAnalyzer(TimeSeriesBinaryClassifier):
    """Class for holding time series predictability prediction.

    Note
    ----
    This class requires ``classification`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self, feature_extractor: BaseTimeSeriesFeatureExtractor, classifier: ClassifierMixin, threshold: float = 0.5
    ):
        """Init PredictabilityAnalyzer with given parameters.

        Parameters
        ----------
        feature_extractor:
            Instance of time series feature extractor.
        classifier:
            Instance of classifier with sklearn interface.
        threshold:
            Positive class probability threshold.
        """
        super().__init__(feature_extractor=feature_extractor, classifier=classifier, threshold=threshold)

    @staticmethod
    def get_series_from_dataset(ts: TSDataset) -> List[np.ndarray]:
        """Transform the dataset into the array with time series samples.

        Series in the result array are sorted in the alphabetical order of the corresponding segment names.

        Parameters
        ----------
        ts:
            TSDataset with the time series.

        Returns
        -------
        :
            Array with time series from TSDataset.
        """
        series = ts[:, sorted(ts.segments), "target"].values.T
        series = [crop_nans_single_series(x) for x in series]
        return series

    def analyze_predictability(self, ts: TSDataset) -> Dict[str, int]:
        """Analyse the time series in the dataset for predictability.

        Parameters
        ----------
        ts:
            Dataset with time series.

        Returns
        -------
        :
            The indicators of predictability for the each segment in the dataset.
        """
        x = self.get_series_from_dataset(ts=ts)
        y_pred = self.predict(x=x)
        result = dict(zip(sorted(ts.segments), y_pred))
        return result

    @staticmethod
    def get_available_models() -> List[str]:
        """Return the list of available models."""
        return ["weasel", "tsfresh", "tsfresh_min"]

    @staticmethod
    def download_model(model_name: str, dataset_freq: str, path: str):
        """Return the list of available models.

        Parameters
        ----------
        model_name:
            Name of the pretrained model.
        dataset_freq:
            Frequency of the dataset.
        path:
            Path to save the file with model.

        Raises
        ------
        ValueError:
            If the model does not exist in s3.
        """
        url = f"http://etna-github-prod.cdn-tinkoff.ru/series_classification/22_11_2022/{dataset_freq}/{model_name}.pickle"
        expected_hash = _KNOWN_MODEL_HASHES.get(url)
        
        # Check if file exists and verify integrity
        if os.path.exists(path):
            if _verify_file_hash(path, expected_hash):
                return  # File exists and is valid (or no hash to check)
            else:
                # File exists but hash doesn't match, re-download
                if expected_hash is not None:
                    warnings.warn(
                        f"Local model file hash does not match expected hash. "
                        f"This may indicate a corrupted download. Re-downloading {model_name} from {url}"
                    )
                os.remove(path)
        
        # Download the file
        try:
            request.urlretrieve(url=url, filename=path)
            
            # Verify the downloaded file
            if not _verify_file_hash(path, expected_hash):
                if expected_hash is not None:
                    os.remove(path)
                    raise RuntimeError(
                        f"Downloaded model file {model_name} from {url} failed integrity check. "
                        f"This may indicate a network issue or corrupted download."
                    )
        except Exception as e:
            if expected_hash is not None and "integrity check" in str(e):
                raise  # Re-raise integrity check errors
            raise ValueError("Model not found! Check the list of available models!")
