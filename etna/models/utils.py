import hashlib
import os
import warnings
from typing import Dict
from typing import Optional
from typing import Union
from urllib import request

import pandas as pd

from etna.datasets.utils import determine_freq  # noqa: F401
from etna.datasets.utils import determine_num_steps  # noqa: F401
from etna.datasets.utils import timestamp_range


# Known model hashes for integrity verification
# To add a hash for a model URL, download the file and compute its MD5 hash
KNOWN_MODEL_HASHES: Dict[str, str] = {
    # Add known model URL -> hash mappings here
    # Example: "http://example.com/model.ckpt": "abcd1234...",
}


def verify_file_hash(file_path: str, expected_hash: Optional[str] = None) -> bool:
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
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
        return file_hash == expected_hash
    except Exception:
        return False


def download_with_integrity_check(
    url: str, 
    destination_path: str, 
    expected_hash: Optional[str] = None,
    force_redownload: bool = False
) -> None:
    """
    Download a file with integrity verification.
    
    Parameters
    ----------
    url:
        URL to download from
    destination_path:
        Local path to save the file
    expected_hash:
        Expected MD5 hash for verification. If None, no verification is performed.
    force_redownload:
        If True, download even if file exists and passes verification
        
    Raises
    ------
    RuntimeError:
        If download fails integrity check
    """
    # Check if file exists and verify integrity
    if os.path.exists(destination_path) and not force_redownload:
        if verify_file_hash(destination_path, expected_hash):
            return  # File exists and is valid
        else:
            # File exists but hash doesn't match, re-download
            if expected_hash is not None:
                warnings.warn(
                    f"Local file hash does not match expected hash. "
                    f"This may indicate a corrupted download. Re-downloading from {url}"
                )
            os.remove(destination_path)

    # Download the file
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    request.urlretrieve(url=url, filename=destination_path)

    # Verify the downloaded file
    if not verify_file_hash(destination_path, expected_hash):
        if expected_hash is not None:
            os.remove(destination_path)
            raise RuntimeError(
                f"Downloaded file from {url} failed integrity check. "
                f"This may indicate a network issue or corrupted download."
            )


def get_known_hash(url: str) -> Optional[str]:
    """
    Get known hash for a URL from the registry.
    
    Parameters
    ----------
    url:
        URL to look up
        
    Returns
    -------
    :
        Known hash for the URL, or None if not found
    """
    return KNOWN_MODEL_HASHES.get(url)


def select_observations(
    df: pd.DataFrame,
    timestamps: pd.Series,
    freq: Union[pd.offsets.BaseOffset, str, None] = None,
    start: Optional[Union[pd.Timestamp, int, str]] = None,
    end: Optional[Union[pd.Timestamp, int, str]] = None,
    periods: Optional[int] = None,
) -> pd.DataFrame:
    """Select observations from dataframe with known timeline.

    Parameters
    ----------
    df:
        dataframe with known timeline
    timestamps:
        series of timestamps to select
    freq:
        frequency of timestamp in df, possible values:

        - :py:class:`pandas.offsets.BaseOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
          for datetime timestamp

        - None for integer timestamp

    start:
        start of the timeline
    end:
        end of the timeline (included)
    periods:
        number of periods in the timeline

    Returns
    -------
    :
        dataframe with selected observations

    Raises
    ------
    ValueError:
        Of the three parameters: start, end, periods, exactly two must be specified
    """
    df["timestamp"] = timestamp_range(start=start, end=end, periods=periods, freq=freq)

    if not (set(timestamps) <= set(df["timestamp"])):
        raise ValueError("Some timestamps do not lie inside the timeline of the provided dataframe.")

    observations = df.set_index("timestamp")
    observations = observations.loc[timestamps]
    observations.reset_index(drop=True, inplace=True)
    return observations
