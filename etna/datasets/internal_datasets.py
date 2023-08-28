import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd

from etna.datasets.tsdataset import TSDataset

_DOWNLOAD_PATH = Path.home() / ".etna"


def _check_dataset_local(dataset_path: Path) -> bool:
    """
    Check dataset is local.

    Parameters
    ----------
    dataset_path:
        path to dataset
    """
    return dataset_path.exists()


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
        with tempfile.TemporaryDirectory() as td:
            temp_path = Path(td) / "temp.zip"
            urllib.request.urlretrieve(url, temp_path)
            with zipfile.ZipFile(temp_path) as f:
                f.extractall(td)
                df = pd.read_csv(Path(td) / file_name, **kwargs)
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
        Whether to rebuild the dataset from the original source. If ``rebuild_dataset=False`` and the dataset was saved
        locally, then it would be loaded from disk. If ``rebuild_dataset=True``, then data

    Returns
    -------
    result:
        internal dataset

    Raises
    ------
    NotImplementedError:
        if name not from available list of dataset names
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
    Download save and load electricity dataset.

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
    dataset_dir.mkdir(exist_ok=True, parents=True)
    data = _download_dataset_zip(url=url, file_name="LD2011_2014.txt", sep=";", dtype=str)
    data = data.rename({"Unnamed: 0": "timestamp"}, axis=1)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.loc[:, data.columns != "timestamp"] = (
        data.loc[:, data.columns != "timestamp"].replace(",", ".", regex=True).astype(float)
    )
    data = data.melt("timestamp", var_name="segment", value_name="target")
    data.to_csv(dataset_dir / "electricity.csv", index=False)
    ts = TSDataset(TSDataset.to_dataset(data), freq="15T")
    return ts


datasets_dict: Dict[str, Dict] = {"electricity": {"get_dataset_function": get_electricity_dataset, "freq": "15T"}}
