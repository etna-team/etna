import tempfile
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from etna.datasets.tsdataset import TSDataset

_DOWNLOAD_PATH = Path.home() / ".etna"
EXOG_SUBDIRECTORY = "exog"


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
        data = pd.read_csv(
            dataset_dir / f"{name}_{parts_[0]}.csv.gz",
            compression="gzip",
            header=[0, 1],
            index_col=[0],
            parse_dates=[0],
        )
        ts = TSDataset(data, freq=freq)
        if _check_dataset_local(dataset_dir / EXOG_SUBDIRECTORY):
            df_exog = pd.read_csv(
                dataset_dir / EXOG_SUBDIRECTORY / f"{name}_{parts_[0]}_exog.csv.gz",
                compression="gzip",
                header=[0, 1],
                index_col=[0],
                parse_dates=[0],
            )
            ts = TSDataset(data, df_exog=df_exog, freq=freq)
        return ts
    else:
        ts_out = []
        for part in parts_:
            data = pd.read_csv(
                dataset_dir / f"{name}_{part}.csv.gz", compression="gzip", header=[0, 1], index_col=[0], parse_dates=[0]
            )
            ts = TSDataset(data, freq=freq)
            if _check_dataset_local(dataset_dir / EXOG_SUBDIRECTORY):
                df_exog = pd.read_csv(
                    dataset_dir / EXOG_SUBDIRECTORY / f"{name}_{part}_exog.csv.gz",
                    compression="gzip",
                    header=[0, 1],
                    index_col=[0],
                    parse_dates=[0],
                )
                ts = TSDataset(data, df_exog=df_exog, freq=freq)
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


def get_m4_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save M4 dataset in different frequency modes.

    The M4 dataset is a collection of 100,000 time series used for the fourth edition of the Makridakis forecasting
    Competition. The M4 dataset consists of time series of yearly, quarterly, monthly and other (weekly, daily and
    hourly) data. Each frequency mode has its own specific prediction horizon: 6 for yearly, 8 for quarterly,
    18 for monthly, 13 for weekly, 14 for daily and 48 for hourly.

    Parameters
    ----------
    dataset_dir:
        The path for saving dataset locally.
    dataset_freq:
        Frequency mode.

    References
    ----------
    .. [1] https://github.com/Mcompetitions/M4-methods
    """
    get_freq = {"Hourly": "H", "Daily": "D", "Weekly": "W-MON", "Monthly": "M", "Quarterly": "QS-OCT", "Yearly": "D"}
    url_data = (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/6c1067e5a57161249b17289a565178dc7a3fb3ca/Dataset/"
    )
    end_date = "2022-01-01"
    freq = get_freq[dataset_freq]

    dataset_dir.mkdir(exist_ok=True, parents=True)

    data_train = pd.read_csv(f"{url_data}/Train/{dataset_freq}-train.csv", index_col=0)
    data_test = pd.read_csv(f"{url_data}/Test/{dataset_freq}-test.csv", index_col=0)

    segments = data_test.index
    test_target = data_test.values

    df_list = []
    test_timestamps = pd.date_range(end=end_date, freq=freq, periods=test_target.shape[1])
    for segment, target in zip(segments, test_target):
        df_segment = pd.DataFrame({"target": target})
        df_segment["segment"] = segment
        df_segment["timestamp"] = test_timestamps
        df_list.append(df_segment)
    df_test = pd.concat(df_list, axis=0)

    train_target = [x[~np.isnan(x)] for x in data_train.values]
    df_list = []
    for segment, target in zip(segments, train_target):
        df_segment = pd.DataFrame({"target": target})
        df_segment["segment"] = segment
        df_segment["timestamp"] = pd.date_range(end=test_timestamps[0], freq=freq, periods=len(target) + 1)[:-1]
        df_list.append(df_segment)
    df_train = pd.concat(df_list, axis=0)

    df_full = pd.concat([df_train, df_test], axis=0)

    TSDataset.to_dataset(df_full).to_csv(
        dataset_dir / f"m4_{dataset_freq.lower()}_full.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / f"m4_{dataset_freq.lower()}_train.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / f"m4_{dataset_freq.lower()}_test.csv.gz", index=True, compression="gzip"
    )


def get_m3_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save M3 dataset in different frequency modes.

    The M3 dataset is a collection of 3,003 time series used for the third edition of the Makridakis forecasting
    Competition. The M3 dataset consists of time series of yearly, quarterly, monthly and other data.
    Each frequency mode has its own specific prediction horizon: 6 for yearly, 8 for quarterly,
    18 for monthly, and 8 for other.

    Parameters
    ----------
    dataset_dir:
        The path for saving dataset locally.
    dataset_freq:
        Frequency mode.

    References
    ----------
    .. [1] https://forvis.github.io/datasets/m3-data/
    .. [2] https://forecasters.org/resources/time-series-data/m3-competition/
    """
    get_freq = {"monthly": "M", "quarterly": "Q-DEC", "yearly": "A-DEC", "other": "Q-DEC"}
    get_horizon = {"monthly": 18, "quarterly": 8, "yearly": 6, "other": 8}
    url_data = "https://forvis.github.io/data"
    end_date = "2022-01-01"
    freq = get_freq[dataset_freq]
    exog_dir = dataset_dir / EXOG_SUBDIRECTORY

    dataset_dir.mkdir(exist_ok=True, parents=True)
    exog_dir.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv(f"{url_data}/M3_{dataset_freq}_TSTS.csv")

    df_full = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    df_full_exog = pd.DataFrame()
    df_test_exog = pd.DataFrame()
    horizon = get_horizon[dataset_freq]
    for _, group in data.groupby("series_id"):
        timestamps = pd.date_range(end=end_date, freq=freq, periods=group.shape[0])
        group.rename(columns={"timestamp": "origin_timestamp", "series_id": "segment", "value": "target"}, inplace=True)
        group["segment"] = group["segment"] + "_" + group["category"]
        group.drop(columns=["category"], inplace=True)
        group["timestamp"] = timestamps

        df_full_part_exog = group.copy()
        df_full_part_exog.drop(columns=["target"], inplace=True)
        group.drop(columns=["origin_timestamp"], inplace=True)

        train_part = group.iloc[:-horizon]
        test_part = group.iloc[-horizon:]
        df_test_part_exog = df_full_part_exog.iloc[-horizon:]

        df_full = pd.concat([df_full, group])
        df_train = pd.concat([df_train, train_part])
        df_test = pd.concat([df_test, test_part])
        df_full_exog = pd.concat([df_full_exog, df_full_part_exog])
        df_test_exog = pd.concat([df_test_exog, df_test_part_exog])

    TSDataset.to_dataset(df_full).to_csv(
        dataset_dir / f"m3_{dataset_freq.lower()}_full.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / f"m3_{dataset_freq.lower()}_train.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / f"m3_{dataset_freq.lower()}_test.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"m3_{dataset_freq.lower()}_full_exog.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"m3_{dataset_freq.lower()}_train_exog.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_test_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"m3_{dataset_freq.lower()}_test_exog.csv.gz", index=True, compression="gzip"
    )


datasets_dict: Dict[str, Dict] = {
    "electricity_15T": {
        "get_dataset_function": get_electricity_dataset_15t,
        "freq": "15T",
        "parts": ("train", "test", "full"),
    },
    "m3_monthly": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="monthly"),
        "freq": "M",
        "parts": ("train", "test", "full"),
    },
    "m3_quarterly": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="quarterly"),
        "freq": "Q-DEC",
        "parts": ("train", "test", "full"),
    },
    "m3_yearly": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="yearly"),
        "freq": "A-DEC",
        "parts": ("train", "test", "full"),
    },
    "m3_other": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="other"),
        "freq": "Q-DEC",
        "parts": ("train", "test", "full"),
    },
    "m4_hourly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Hourly"),
        "freq": "H",
        "parts": ("train", "test", "full"),
    },
    "m4_daily": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Daily"),
        "freq": "D",
        "parts": ("train", "test", "full"),
    },
    "m4_weekly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Weekly"),
        "freq": "W-MON",
        "parts": ("train", "test", "full"),
    },
    "m4_monthly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Monthly"),
        "freq": "M",
        "parts": ("train", "test", "full"),
    },
    "m4_quarterly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Quarterly"),
        "freq": "QS-JAN",
        "parts": ("train", "test", "full"),
    },
    "m4_yearly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Yearly"),
        "freq": "D",
        "parts": ("train", "test", "full"),
    },
}
