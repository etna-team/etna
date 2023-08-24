import os
import tempfile
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict
from typing import Callable

import pandas as pd
import requests

from etna.datasets import TSDataset

_DOWNLOAD_PATH = Path.home() / ".etna"


def _check_dataset_local(dataset_path: Path) -> bool:
    """
    Check dataset is local.

    Parameters
    ----------
    dataset_path:
        path to dataset
    """
    return os.path.isfile(dataset_path)


def _download_dataset_zip(url: str, file_name: str, **kwargs) -> pd.DataFrame:
    """
    Download zipped csv file.

    Parameters
    ----------
    url:
        url of the dataset
    file_name:
        csv file name in zip archive

    Returns
    -------
    result:
        dataframe with data

    Raises
    ------
    Exception:
        any error during downloading, saving and reading dataset from url
    """
    try:
        req = requests.get(url)
        with zipfile.ZipFile(BytesIO(req.content)) as f:
            with tempfile.TemporaryDirectory() as td:
                f.extractall(td)
                df = pd.read_csv(os.path.join(td, file_name), **kwargs)
    except Exception as err:
        raise Exception(f"Error during downloading and reading dataset. Reason: {repr(err)}")
    return df


def load_dataset(name: str, download_path: Path = _DOWNLOAD_PATH, rebuild_dataset: bool = False) -> TSDataset:
    """
    Load internal dataset.

    Parameters
    ----------
    name:
        Name of the dataset.
    download_path:
        The path for saving dataset locally.
    rebuild_dataset:
        Whether to rebuild dataset from the initial source.

    Returns
    -------
    result:
        internal dataset
    """
    if name not in datasets_dict:
        raise NotImplementedError(f"Dataset {name} is not available. You can use one from: {sorted(datasets_dict)}.")

    dataset_dir = download_path / name
    dataset_path = dataset_dir / f"{name}.csv"

    get_dataset_function, freq = datasets_dict[name].values()
    if not _check_dataset_local(dataset_path) or rebuild_dataset:
        ts = get_dataset_function(dataset_dir)
    else:
        data = pd.read_csv(dataset_path)
        ts = TSDataset(TSDataset.to_dataset(data), freq=freq)
    return ts


def get_electricity_dataset(dataset_dir) -> TSDataset:
    """
    Download6 save and load electricity dataset.
    The electricity dataset is a 15 minutes time series of electricity consumption (in kW)
    of 370 customers.

    Parameters
    ----------
    dataset_dir:
        The path for saving dataset locally.

    Returns
    -------
    result:
        electricity dataset in TSDataset format

    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
    """
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    os.makedirs(dataset_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _download_dataset_zip(url=url, file_name="LD2011_2014.txt", sep=";")
    data = data.rename({"Unnamed: 0": "timestamp"}, axis=1)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.loc[:, data.columns != "timestamp"] = (
        data.loc[:, data.columns != "timestamp"].replace(",", ".", regex=True).astype(float)
    )
    data = data.melt("timestamp", var_name="segment", value_name="target")
    data.to_csv(dataset_dir / "electricity.csv", index=False)
    ts = TSDataset(TSDataset.to_dataset(data), freq="15T")
    return ts


datasets_dict: Dict[str: Dict[Callable, str]] = {"electricity": {"get_dataset_function": get_electricity_dataset, "freq": "15T"}}
