import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

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


def load_dataset(
    name: str,
    download_path: Path = _DOWNLOAD_PATH,
    rebuild_dataset: bool = False,
    parts: Union[str, Tuple[str, ...]] = "full",
) -> Union[TSDataset, List[TSDataset]]:
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
        locally, then it would be loaded from disk. If ``rebuild_dataset=True``, then the dataset will be downloaded and
        saved locally.
    parts:
        Parts of the dataset to load. Each dataset has specific parts (e.g. ``("train", "test", "full")`` for
        ``electricity_15T`` dataset). By default, all datasets have "full" part, other parts may vary.

        - If parts is str, then the function will return a single ``TSDataset`` object.
        - If parts is a tuple of multiple elements, then the function will return a list of ``TSDataset`` objects.

    Returns
    -------
    result:
        internal dataset

    Raises
    ------
    NotImplementedError:
        if name not from available list of dataset names
    NotImplementedError:
        if part not from available list of dataset parts
    """
    if name not in datasets_dict:
        raise NotImplementedError(f"Dataset {name} is not available. You can use one from: {sorted(datasets_dict)}.")

    parts_ = (parts,) if isinstance(parts, str) else parts
    dataset_params = datasets_dict[name]
    for part in parts_:
        if part not in dataset_params["parts"]:
            raise NotImplementedError(f"Part {part} is not available. You can use one from: {dataset_params['parts']}.")

    dataset_dir = download_path / name
    dataset_path = dataset_dir / f"{name}_full.csv.gz"

    get_dataset_function = dataset_params["get_dataset_function"]
    freq = dataset_params["freq"]

    if not _check_dataset_local(dataset_path) or rebuild_dataset:
        get_dataset_function(dataset_dir)

    if len(parts_) == 1:
        data = pd.read_csv(dataset_dir / f"{name}_{parts_[0]}.csv.gz", compression="gzip", header=[0, 1], index_col=[0])
        ts = TSDataset(data, freq=freq)
        return ts
    else:
        ts_out = []
        for part in parts_:
            data = pd.read_csv(dataset_dir / f"{name}_{part}.csv.gz", compression="gzip", header=[0, 1], index_col=[0])
            ts = TSDataset(data, freq=freq)
            ts_out.append(ts)
        return ts_out


def get_electricity_dataset_15t(dataset_dir) -> None:
    """
    Download and save electricity dataset in three parts: full, train, test.

    The electricity dataset is a 15 minutes time series of electricity consumption (in kW)
    of 370 customers.

    Parameters
    ----------
    dataset_dir:
        The path for saving dataset locally.

    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
    """
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    dataset_dir.mkdir(exist_ok=True, parents=True)
    data = _download_dataset_zip(url=url, file_name="LD2011_2014.txt", sep=";", dtype=str)

    data = data.rename({"Unnamed: 0": "timestamp"}, axis=1)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    dt_list = sorted(data["timestamp"].unique())
    data = data.melt("timestamp", var_name="segment", value_name="target")
    data["target"] = data["target"].str.replace(",", ".").astype(float)

    data_train = data[data["timestamp"].isin(dt_list[: -15 * 24])]
    data_test = data[data["timestamp"].isin(dt_list[-15 * 24 :])]
    TSDataset.to_dataset(data).to_csv(dataset_dir / "electricity_15T_full.csv.gz", index=True, compression="gzip")
    TSDataset.to_dataset(data_train).to_csv(
        dataset_dir / "electricity_15T_train.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(data_test).to_csv(dataset_dir / "electricity_15T_test.csv.gz", index=True, compression="gzip")


datasets_dict: Dict[str, Dict] = {
    "electricity_15T": {
        "get_dataset_function": get_electricity_dataset_15t,
        "freq": "15T",
        "parts": ("train", "test", "full"),
    }
}
