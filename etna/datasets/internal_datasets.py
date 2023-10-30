import tempfile
import urllib.request
import zipfile
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import holidays
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


def _download_dataset_zip(
    url: str, file_names: Union[str, Tuple[str, ...]], read_functions: Union[Callable, Tuple[Callable, ...]]
) -> Any:
    """
    Download zipped files.

    Parameters
    ----------
    url:
        url of the dataset
    file_names:
        file names in zip archive to load
    read_functions:
        functions for loading files from zip archive

    Returns
    -------
    result:
        data from zip archive

    Raises
    ------
    Exception:
        any error during downloading, saving and reading dataset from url
    """
    file_names_ = (file_names,) if isinstance(file_names, str) else file_names
    read_functions_ = (read_functions,) if callable(read_functions) else read_functions
    try:
        with tempfile.TemporaryDirectory() as td:
            temp_path = Path(td) / "temp.zip"
            urllib.request.urlretrieve(url, temp_path)
            with zipfile.ZipFile(temp_path) as f:
                f.extractall(td)
                out = []
                for file_name, read_function in zip(file_names_, read_functions_):
                    data = read_function(Path(td) / file_name)
                    out.append(data)
                out = out[0] if len(out) == 1 else out
    except Exception as err:
        raise Exception(f"Error during downloading and reading dataset. Reason: {repr(err)}")
    return out


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
        if _check_dataset_local(dataset_dir / EXOG_SUBDIRECTORY):
            df_exog = pd.read_csv(
                dataset_dir / EXOG_SUBDIRECTORY / f"{name}_{parts_[0]}_exog.csv.gz",
                compression="gzip",
                header=[0, 1],
                index_col=[0],
                parse_dates=[0],
            )
            ts = TSDataset(data, df_exog=df_exog, freq=freq)
        else:
            ts = TSDataset(data, freq=freq)
        return ts
    else:
        ts_out = []
        for part in parts_:
            data = pd.read_csv(
                dataset_dir / f"{name}_{part}.csv.gz", compression="gzip", header=[0, 1], index_col=[0], parse_dates=[0]
            )
            if _check_dataset_local(dataset_dir / EXOG_SUBDIRECTORY):
                df_exog = pd.read_csv(
                    dataset_dir / EXOG_SUBDIRECTORY / f"{name}_{part}_exog.csv.gz",
                    compression="gzip",
                    header=[0, 1],
                    index_col=[0],
                    parse_dates=[0],
                )
                ts = TSDataset(data, df_exog=df_exog, freq=freq)
            else:
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
    data = _download_dataset_zip(
        url=url, file_names="LD2011_2014.txt", read_functions=partial(pd.read_csv, sep=";", dtype=str)
    )
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


def get_traffic_2008_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save traffic (2008-2009) dataset in different frequency modes.

    15 months worth of daily data (440 daily records) that describes the occupancy rate, between 0 and 1, of different
    car lanes of the San Francisco bay area freeways across time. Data was collected by 963 sensors from
    Jan. 1st 2008 to Mar. 30th 2009 (15 days were dropped from this period: public holidays and two days with
    anomalies, we set zero values for these days). Initial dataset has 10 min frequency, we create traffic with hour
    frequency by mean aggregation. Each frequency mode has its own specific prediction horizon: 6 * 24 for 10T,
    24 for hourly.

    Notes
    -----
    There is another "traffic" dataset that is also popular and used in papers for time series tasks. This
    dataset is also from the California Department of Transportation PEMS website, http://pems.dot.ca.gov, however for
    different time period: from 2015 to 2016. We also have it in our library ("traffic_2015").

    References
    ----------
    .. [1] https://archive.ics.uci.edu/dataset/204/pems+sf
    .. [2] http://pems.dot.ca.gov
    """

    def read_data(path: Path, part: str) -> np.ndarray:
        with open(path, "r") as f:
            if part in ("randperm", "stations_list"):
                data = f.read().lstrip("[").rstrip("]\n").split(" ")
                out = np.array(map(int, data)) if part == "randperm" else np.array(data)
                return out
            else:
                lines = []
                for line in f:
                    line_segments = line.lstrip("[").rstrip("]\n").split(";")
                    line_target = [list(map(float, segment.split(" "))) for segment in line_segments]
                    lines.append(line_target)
                out = np.array(lines)
                return out

    url = "https://archive.ics.uci.edu/static/public/204/pems+sf.zip"

    dataset_dir.mkdir(exist_ok=True, parents=True)

    file_names = ("randperm", "stations_list", "PEMS_train", "PEMS_test")
    read_functions = tuple(partial(read_data, part=file_name) for file_name in file_names)

    ts_indecies, stations, targets_train, targets_test = _download_dataset_zip(
        url=url, file_names=file_names, read_functions=read_functions
    )

    targets = np.concatenate([targets_train, targets_test], axis=0)
    targets = targets[np.argsort(ts_indecies)].reshape(-1, 963)

    drop_days = (
        list(holidays.country_holidays(country="US", years=2008).keys())
        + list(holidays.country_holidays(country="US", years=2009).keys())[:3]
        + [date(2009, 3, 8), date(2009, 3, 10)]
    )

    dates_df = pd.DataFrame({"timestamp": pd.date_range("2008-01-01 00:00:00", "2009-03-30 23:50:00", freq="10T")})
    dates_df["dt"] = dates_df["timestamp"].dt.date
    dates_df_cropped = dates_df[~dates_df["dt"].isin(drop_days)]
    dates_df = dates_df.drop(["dt"], axis=1)

    df = pd.DataFrame(targets, columns=stations)
    df["timestamp"] = dates_df_cropped["timestamp"].values
    df = df.merge(dates_df, on=["timestamp"], how="right").fillna(0)
    df = df.melt("timestamp", var_name="segment", value_name="target")

    if dataset_freq == "10T":
        df_full = TSDataset.to_dataset(df)
        df_test = df_full.tail(6 * 24)
        df_train = df_full[~df_full.index.isin(df_test.index)]
    elif dataset_freq == "hourly":
        df["timestamp"] = df["timestamp"].dt.floor("h")
        df = df.groupby(["timestamp", "segment"], as_index=False)[["target"]].mean()
        df_full = TSDataset.to_dataset(df)
        df_test = df_full.tail(24)
        df_train = df_full[~df_full.index.isin(df_test.index)]
    else:
        raise NotImplementedError(f"traffic_2008 with {dataset_freq} frequency is not available.")

    df_full.to_csv(dataset_dir / f"traffic_2008_{dataset_freq.lower()}_full.csv.gz", index=True, compression="gzip")
    df_train.to_csv(dataset_dir / f"traffic_2008_{dataset_freq.lower()}_train.csv.gz", index=True, compression="gzip")
    df_test.to_csv(dataset_dir / f"traffic_2008_{dataset_freq.lower()}_test.csv.gz", index=True, compression="gzip")


def get_traffic_2015_dataset(dataset_dir: Path) -> None:
    """
    Download and save traffic (2015-2016) dataset.

    24 months worth of hourly data (24 daily records) that describes the occupancy rate, between 0 and 1, of different
    car lanes of the San Francisco bay area freeways across time. Data was collected by 862 sensors from
    Jan. 1st 2015 to Dec. 31th 2016. Dataset has prediction horizon: 24.

    Notes
    -----
    There is another "traffic" dataset that is also popular and used in papers for time series tasks. This
    dataset is also from the California Department of Transportation PEMS website, http://pems.dot.ca.gov, however for
    different time period: from 2008 to 2009. We also have it in our library ("traffic_2008").

    References
    ----------
    .. [1] https://github.com/laiguokun/multivariate-time-series-data
    .. [2] http://pems.dot.ca.gov
    """
    url = (
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
        "7f402f185cc2435b5e66aed13a3b560ed142e023/traffic/traffic.txt.gz"
    )

    dataset_dir.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv(url, header=None)
    timestamps = pd.date_range("2015-01-01", freq="H", periods=data.shape[0])
    data["timestamp"] = timestamps
    data = data.melt("timestamp", var_name="segment", value_name="target")

    df_full = TSDataset.to_dataset(data)
    df_test = df_full.tail(24)
    df_train = df_full[~df_full.index.isin(df_test.index)]

    df_full.to_csv(dataset_dir / f"traffic_2015_hourly_full.csv.gz", index=True, compression="gzip")
    df_train.to_csv(dataset_dir / f"traffic_2015_hourly_train.csv.gz", index=True, compression="gzip")
    df_test.to_csv(dataset_dir / f"traffic_2015_hourly_test.csv.gz", index=True, compression="gzip")


def get_m3_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save M3 dataset in different frequency modes.

    The M3 dataset is a collection of 3,003 time series used for the third edition of the Makridakis forecasting
    Competition. The M3 dataset consists of time series of yearly, quarterly, monthly and other data. Dataset with other
    data originally does not have any particular frequency, but we assume it as a quarterly data. Each frequency mode
    has its own specific prediction horizon: 6 for yearly, 8 for quarterly, 18 for monthly, and 8 for other.

    M3 dataset has series ending on different dates. As to the specificity of TSDataset we should add custom dates
    to make series end on one date. Original dates are added as an exogenous data. For example, ``df_exog`` of train
    dataset has dates for train and test and ``df_exog`` of test dataset has dates only for test.

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


def get_tourism_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save tourism dataset in different frequency modes.

    Dataset contains 1311 series in three frequency modes: monthly, quarterly, yearly. They were supplied by both
    tourism bodies (such as Tourism Australia, the Hong Kong Tourism Board and Tourism New Zealand) and various
    academics, who had used them in previous tourism forecasting studies. Each frequency mode has its own specific
    prediction horizon: 4 for yearly, 8 for quarterly, 24 for monthly.

    Tourism dataset has series ending on different dates. As to the specificity of TSDataset we should add custom dates
    to make series end on one date. Original dates are added as an exogenous data. For example, ``df_exog`` of train
    dataset has dates for train and test and ``df_exog`` of test dataset has dates only for test.

    References
    ----------
    .. [1] https://robjhyndman.com/publications/the-tourism-forecasting-competition/
    """
    get_freq = {"monthly": "M", "quarterly": "Q-DEC", "yearly": "A-DEC"}
    start_index_target_rows = {"monthly": 3, "quarterly": 3, "yearly": 2}
    end_date = "2022-01-01"
    freq = get_freq[dataset_freq]
    target_index = start_index_target_rows[dataset_freq]
    exog_dir = dataset_dir / EXOG_SUBDIRECTORY

    exog_dir.mkdir(exist_ok=True, parents=True)

    data_train, data_test = _download_dataset_zip(
        "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip",
        file_names=(f"{dataset_freq}_in.csv", f"{dataset_freq}_oos.csv"),
        read_functions=(partial(pd.read_csv, sep=","), partial(pd.read_csv, sep=",")),
    )

    segments = data_train.columns

    df_full = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    df_full_exog = pd.DataFrame()
    df_test_exog = pd.DataFrame()
    for seg in segments:
        data_train_ = data_train[seg].values
        data_test_ = data_test[seg].values

        train_size = int(data_train_[0])
        test_size = int(data_test_[0])

        date_params = list(map(int, data_train_[~np.isnan(data_train_)][1:target_index]))
        initial_date = date(date_params[0], date_params[1], 1) if len(date_params) == 2 else date(date_params[0], 1, 1)

        target_train = data_train_[~np.isnan(data_train_)][target_index : target_index + train_size]
        target_test = data_test_[target_index : target_index + test_size]
        target_full = np.concatenate([target_train, target_test])

        new_timestamps = pd.date_range(end=end_date, freq=freq, periods=len(target_full))
        initial_timestamps = pd.date_range(start=initial_date, periods=len(target_full), freq=freq)

        df_full_ = pd.DataFrame(
            {"timestamp": new_timestamps, "segment": [seg] * len(target_full), "target": target_full}
        )
        df_train_ = df_full_.head(train_size)
        df_test_ = df_full_.tail(test_size)

        df_full_exog_ = pd.DataFrame(
            {"timestamp": new_timestamps, "segment": [seg] * len(target_full), "target": initial_timestamps}
        )
        df_test_exog_ = df_full_exog_.tail(test_size)

        df_full = pd.concat([df_full, df_full_])
        df_train = pd.concat([df_train, df_train_])
        df_test = pd.concat([df_test, df_test_])
        df_full_exog = pd.concat([df_full_exog, df_full_exog_])
        df_test_exog = pd.concat([df_test_exog, df_test_exog_])

    TSDataset.to_dataset(df_full).to_csv(
        dataset_dir / f"tourism_{dataset_freq.lower()}_full.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / f"tourism_{dataset_freq.lower()}_train.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / f"tourism_{dataset_freq.lower()}_test.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"tourism_{dataset_freq.lower()}_full_exog.csv.gz",
        index=True,
        compression="gzip",
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"tourism_{dataset_freq.lower()}_train_exog.csv.gz",
        index=True,
        compression="gzip",
    )
    TSDataset.to_dataset(df_test_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"tourism_{dataset_freq.lower()}_test_exog.csv.gz",
        index=True,
        compression="gzip",
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
    "traffic_2008_10T": {
        "get_dataset_function": partial(get_traffic_2008_dataset, dataset_freq="10T"),
        "freq": "10T",
        "parts": ("train", "test", "full"),
    },
    "traffic_2008_hourly": {
        "get_dataset_function": partial(get_traffic_2008_dataset, dataset_freq="hourly"),
        "freq": "H",
        "parts": ("train", "test", "full"),
    },
    "traffic_2015_hourly": {
        "get_dataset_function": get_traffic_2015_dataset,
        "freq": "H",
        "parts": ("train", "test", "full"),
    },
    "tourism_monthly": {
        "get_dataset_function": partial(get_tourism_dataset, dataset_freq="monthly"),
        "freq": "MS",
        "parts": ("train", "test", "full"),
    },
    "tourism_quarterly": {
        "get_dataset_function": partial(get_tourism_dataset, dataset_freq="quarterly"),
        "freq": "Q-DEC",
        "parts": ("train", "test", "full"),
    },
    "tourism_yearly": {
        "get_dataset_function": partial(get_tourism_dataset, dataset_freq="yearly"),
        "freq": "A-DEC",
        "parts": ("train", "test", "full"),
    },
}
