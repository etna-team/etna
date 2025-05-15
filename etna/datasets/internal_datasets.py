import gzip
import hashlib
import tempfile
import urllib.request
import warnings
import zipfile
from datetime import date
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from etna.datasets.tsdataset import TSDataset

_DOWNLOAD_PATH = Path.home() / ".etna" / "internal_datasets"
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


def read_dataset(dataset_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Read locally saved dataset in bytes, calculate hash and build ``pandas.DataFrame``.

    Parameters
    ----------
    dataset_path:
        The path of dataset.

    Returns
    -------
    result:
        dataset, hash
    """
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        data_ = f.read()

    h = hashlib.md5(data_.encode()).hexdigest()
    data = pd.read_csv(StringIO(data_), header=[0, 1], index_col=[0], parse_dates=[0])
    return data, h


def load_dataset(
    name: str,
    download_path: Path = _DOWNLOAD_PATH,
    rebuild_dataset: bool = False,
    parts: Union[str, Tuple[str, ...]] = "full",
) -> Union[TSDataset, List[TSDataset]]:
    """
    Load internal dataset. Full list of available datasets you can see on :ref:`internal datasets page <internal_datasets>`.

    Parameters
    ----------
    name:
        Name of the dataset.
    download_path:
        The path for saving dataset locally. By default it is directory "~/.etna/internal_datasets".
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
    ts_out = []
    for part in parts_:
        data, dataset_hash = read_dataset(dataset_path=dataset_dir / f"{name}_{part}.csv.gz")
        if dataset_hash != datasets_dict[name]["hash"][part]:
            warnings.warn(
                f"Local hash and expected hash are different for {name} record part {part}. "
                "The first possible reason is that the local copy of the dataset is out of date. In this case you can "
                "try setting rebuild_dataset=True to rebuild the dataset. The second possible reason is that the local "
                "copy of the dataset reflects a more recent version of the data than your version of the library. "
                "In this case you can try updating the library version."
            )
        if _check_dataset_local(dataset_dir / EXOG_SUBDIRECTORY):
            df_exog = pd.read_csv(
                dataset_dir / EXOG_SUBDIRECTORY / f"{name}_{part}_exog.csv.gz",
                compression="gzip",
                header=[0, 1],
                index_col=[0],
                parse_dates=[0],
            )
            # For some datasets there are real dates that we cannot use directly, so we save them in exog data. When we
            # load dataset, we convert this dates into datetime so that the user can apply transforms to them.
            if "exog_datetime_columns" in dataset_params:
                dt_columns = [col for col in df_exog.columns if col[1] in dataset_params["exog_datetime_columns"]]
                df_exog[dt_columns] = df_exog[dt_columns].astype("datetime64[ns]")
            ts = TSDataset(data, df_exog=df_exog, freq=freq)
        else:
            ts = TSDataset(data, freq=freq)
        ts_out.append(ts)

    if len(ts_out) == 1:
        return ts_out[0]
    else:
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
    url_data = (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/6c1067e5a57161249b17289a565178dc7a3fb3ca/Dataset/"
    )

    dataset_dir.mkdir(exist_ok=True, parents=True)

    data_train = pd.read_csv(f"{url_data}/Train/{dataset_freq}-train.csv", index_col=0)
    data_test = pd.read_csv(f"{url_data}/Test/{dataset_freq}-test.csv", index_col=0)

    segments = data_test.index
    test_target = data_test.values
    test_len = test_target.shape[1]
    train_target = [x[~np.isnan(x)] for x in data_train.values]

    max_len = test_len + max([len(target) for target in train_target])

    df_list = []
    test_timestamps = np.arange(start=max_len - test_len, stop=max_len)
    for segment, target in zip(segments, test_target):
        df_segment = pd.DataFrame({"target": target})
        df_segment["segment"] = segment
        df_segment["timestamp"] = test_timestamps
        df_list.append(df_segment)
    df_test = pd.concat(df_list, axis=0)

    df_list = []
    for segment, target in zip(segments, train_target):
        df_segment = pd.DataFrame({"target": target})
        df_segment["segment"] = segment
        df_segment["timestamp"] = np.arange(start=max_len - test_target.shape[1] - len(target), stop=max_len - test_len)
        df_list.append(df_segment)
    df_train = pd.concat(df_list, axis=0)

    df_full = pd.concat([df_train, df_test], axis=0)

    TSDataset.to_dataset(df_full).to_csv(
        dataset_dir / f"m4_{dataset_freq.lower()}_full.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / f"m4_{dataset_freq.lower()}_train.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / f"m4_{dataset_freq.lower()}_test.csv.gz", index=True, compression="gzip", float_format="%.8f"
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
                out = np.array(list(map(int, data))) if part == "randperm" else np.array(data)
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

    # federal holidays and days with anomalies
    drop_days = [
        date(2008, 1, 1),
        date(2008, 1, 21),
        date(2008, 2, 18),
        date(2008, 5, 26),
        date(2008, 7, 4),
        date(2008, 9, 1),
        date(2008, 10, 13),
        date(2008, 11, 11),
        date(2008, 11, 27),
        date(2008, 12, 25),
        date(2009, 1, 1),
        date(2009, 1, 19),
        date(2009, 2, 16),
    ] + [date(2008, 3, 8), date(2009, 3, 9)]

    dates_df = pd.DataFrame(
        {"timestamp": pd.date_range("2008-01-01 00:00:00", "2009-03-30 23:50:00", freq=pd.offsets.Minute(n=10))}
    )
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

    df_full.to_csv(
        dataset_dir / f"traffic_2008_{dataset_freq.lower()}_full.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    df_train.to_csv(
        dataset_dir / f"traffic_2008_{dataset_freq.lower()}_train.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    df_test.to_csv(
        dataset_dir / f"traffic_2008_{dataset_freq.lower()}_test.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )


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
    timestamps = pd.date_range("2015-01-01", freq=pd.offsets.Hour(), periods=data.shape[0])
    data["timestamp"] = timestamps
    data = data.melt("timestamp", var_name="segment", value_name="target")

    df_full = TSDataset.to_dataset(data)
    df_test = df_full.tail(24)
    df_train = df_full[~df_full.index.isin(df_test.index)]

    df_full.to_csv(
        dataset_dir / f"traffic_2015_hourly_full.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )
    df_train.to_csv(
        dataset_dir / f"traffic_2015_hourly_train.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )
    df_test.to_csv(
        dataset_dir / f"traffic_2015_hourly_test.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )


def get_m3_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save M3 dataset in different frequency modes.

    The M3 dataset is a collection of 3,003 time series used for the third edition of the Makridakis forecasting
    Competition. The M3 dataset consists of time series of yearly, quarterly, monthly and other data. Dataset with other
    data originally does not have any particular frequency, but we assume it as a quarterly data. Each frequency mode
    has its own specific prediction horizon: 6 for yearly, 8 for quarterly, 18 for monthly, and 8 for other.

    M3 dataset has series ending on different dates. As to the specificity of TSDataset we use integer index use integer
    index to make series end on one timestamp. Original dates are added as an exogenous data. For example, ``df_exog``
    of train dataset has dates for train and test and ``df_exog`` of test dataset has dates only for test.

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
    get_horizon = {"monthly": 18, "quarterly": 8, "yearly": 6, "other": 8}
    url_data = "https://forvis.github.io/data"
    horizon = get_horizon[dataset_freq]
    exog_dir = dataset_dir / EXOG_SUBDIRECTORY

    exog_dir.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv(f"{url_data}/M3_{dataset_freq}_TSTS.csv")
    max_len = data.groupby("series_id")["timestamp"].count().max()

    df_full = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    df_full_exog = pd.DataFrame()
    df_test_exog = pd.DataFrame()
    for _, group in data.groupby("series_id"):
        timestamps = np.arange(start=max_len - group.shape[0], stop=max_len)
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

    if dataset_freq == "yearly":
        df_full_exog["origin_timestamp"] = pd.to_datetime(df_full_exog["origin_timestamp"], format="%Y")
        df_test_exog["origin_timestamp"] = pd.to_datetime(df_test_exog["origin_timestamp"], format="%Y")
    elif dataset_freq != "other":
        df_full_exog["origin_timestamp"] = pd.to_datetime(df_full_exog["origin_timestamp"])
        df_test_exog["origin_timestamp"] = pd.to_datetime(df_test_exog["origin_timestamp"])

    TSDataset.to_dataset(df_full).to_csv(
        dataset_dir / f"m3_{dataset_freq.lower()}_full.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / f"m3_{dataset_freq.lower()}_train.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / f"m3_{dataset_freq.lower()}_test.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )

    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"m3_{dataset_freq.lower()}_full_exog.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"m3_{dataset_freq.lower()}_train_exog.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_test_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"m3_{dataset_freq.lower()}_test_exog.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )


def get_tourism_dataset(dataset_dir: Path, dataset_freq: str) -> None:
    """
    Download and save tourism dataset in different frequency modes.

    Dataset contains 1311 series in three frequency modes: monthly, quarterly, yearly. They were supplied by both
    tourism bodies (such as Tourism Australia, the Hong Kong Tourism Board and Tourism New Zealand) and various
    academics, who had used them in previous tourism forecasting studies. Each frequency mode has its own specific
    prediction horizon: 4 for yearly, 8 for quarterly, 24 for monthly.

    Tourism dataset has series ending on different dates. As to the specificity of TSDataset we use integer index to
    make series end on one timestamp. Original dates are added as an exogenous data. For example, ``df_exog`` of train
    dataset has dates for train and test and ``df_exog`` of test dataset has dates only for test.

    References
    ----------
    .. [1] https://robjhyndman.com/publications/the-tourism-forecasting-competition/
    """
    get_freq = {
        "monthly": pd.offsets.MonthBegin(),
        "quarterly": pd.offsets.QuarterEnd(startingMonth=12),
        "yearly": pd.offsets.YearEnd(),
    }
    start_index_target_rows = {"monthly": 3, "quarterly": 3, "yearly": 2}
    freq = get_freq[dataset_freq]
    target_index = start_index_target_rows[dataset_freq]
    exog_dir = dataset_dir / EXOG_SUBDIRECTORY

    exog_dir.mkdir(exist_ok=True, parents=True)

    data_train, data_test = _download_dataset_zip(
        "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip",
        file_names=(f"{dataset_freq}_in.csv", f"{dataset_freq}_oos.csv"),
        read_functions=(partial(pd.read_csv, sep=","), partial(pd.read_csv, sep=",")),
    )
    max_len = int(data_train.iloc[0].max() + data_test.iloc[0].max())
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

        new_timestamps = np.arange(start=max_len - len(target_full), stop=max_len)
        initial_timestamps = pd.date_range(start=initial_date, periods=len(target_full), freq=freq)

        df_full_ = pd.DataFrame(
            {"timestamp": new_timestamps, "segment": [seg] * len(target_full), "target": target_full}
        )
        df_train_ = df_full_.head(train_size)
        df_test_ = df_full_.tail(test_size)

        df_full_exog_ = pd.DataFrame(
            {"timestamp": new_timestamps, "segment": [seg] * len(target_full), "origin_timestamp": initial_timestamps}
        )
        df_test_exog_ = df_full_exog_.tail(test_size)

        df_full = pd.concat([df_full, df_full_])
        df_train = pd.concat([df_train, df_train_])
        df_test = pd.concat([df_test, df_test_])
        df_full_exog = pd.concat([df_full_exog, df_full_exog_])
        df_test_exog = pd.concat([df_test_exog, df_test_exog_])

    TSDataset.to_dataset(df_full).to_csv(
        dataset_dir / f"tourism_{dataset_freq.lower()}_full.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / f"tourism_{dataset_freq.lower()}_train.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / f"tourism_{dataset_freq.lower()}_test.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"tourism_{dataset_freq.lower()}_full_exog.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_full_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"tourism_{dataset_freq.lower()}_train_exog.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )
    TSDataset.to_dataset(df_test_exog).to_csv(
        dataset_dir / EXOG_SUBDIRECTORY / f"tourism_{dataset_freq.lower()}_test_exog.csv.gz",
        index=True,
        compression="gzip",
        float_format="%.8f",
    )


def get_weather_dataset(dataset_dir: Path) -> None:
    """
    Download and save weather dataset.

    Dataset contains 21 meteorological indicators in Germany, such as humidity and air temperature with a 10 min
    frequency for 2020. We use the last 24 hours as prediction horizon.

    References
    ----------
    .. [1] https://www.bgc-jena.mpg.de/wetter/
    """
    url = "https://www.bgc-jena.mpg.de/wetter/{dataset_name}.zip"

    dataset_dir.mkdir(exist_ok=True, parents=True)

    data = pd.DataFrame()
    for dataset_name in ("mpi_roof_2020a", "mpi_roof_2020b"):
        data_ = _download_dataset_zip(
            url.format(dataset_name=dataset_name),
            file_names=dataset_name + ".csv",
            read_functions=partial(pd.read_csv, encoding="cp1252"),
        ).drop_duplicates(subset=["Date Time"])
        data = pd.concat([data, data_])

    data = data.rename({"Date Time": "timestamp"}, axis=1)
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d.%m.%Y %H:%M:%S")

    data = data.melt("timestamp", var_name="segment", value_name="target")

    df_full = TSDataset.to_dataset(data)
    df_test = df_full.tail(6 * 24)
    df_train = df_full[~df_full.index.isin(df_test.index)]

    df_full.to_csv(dataset_dir / f"weather_10T_full.csv.gz", index=True, compression="gzip", float_format="%.8f")
    df_train.to_csv(dataset_dir / f"weather_10T_train.csv.gz", index=True, compression="gzip", float_format="%.8f")
    df_test.to_csv(dataset_dir / f"weather_10T_test.csv.gz", index=True, compression="gzip", float_format="%.8f")


def get_ett_dataset(dataset_dir: Path, dataset_type: str) -> None:
    """
    Download and save Electricity Transformer Datasets (small version).

    Dataset consists of four parts: ETTh1 (hourly freq), ETTh2 (hourly freq), ETTm1 (15 min freq), ETTm2 (15 min freq).
    This dataset is a collection of two years of data from two regions of a province of China. There are one target
    column ("oil temperature") and six different types of external power load features. We use the last 720 hours as
    prediction horizon.

    References
    ----------
    .. [1] https://www.bgc-jena.mpg.de/wetter/
    .. [2] https://arxiv.org/abs/2012.07436
    """
    url = (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
        "1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/{name}.csv"
    )
    dataset_dir.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv(url.format(name=dataset_type))
    data = data.rename({"date": "timestamp"}, axis=1)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.melt("timestamp", var_name="segment", value_name="target")

    df_full = TSDataset.to_dataset(data)
    if dataset_type in ("ETTm1", "ETTm2"):
        df_test = df_full.tail(720 * 4)
        df_train = df_full.head(len(df_full) - 720 * 4)
    elif dataset_type in ("ETTh1", "ETTh2"):
        df_test = df_full.tail(720)
        df_train = df_full.head(len(df_full) - 720)
    else:
        raise NotImplementedError(
            f"ETT dataset does not have '{dataset_type}' dataset_type."
            f"You can use one from: ('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2')."
        )

    df_full.to_csv(dataset_dir / f"{dataset_type}_full.csv.gz", index=True, compression="gzip", float_format="%.8f")
    df_train.to_csv(dataset_dir / f"{dataset_type}_train.csv.gz", index=True, compression="gzip", float_format="%.8f")
    df_test.to_csv(dataset_dir / f"{dataset_type}_test.csv.gz", index=True, compression="gzip", float_format="%.8f")


def get_ihepc_dataset(dataset_dir: Path) -> None:
    """
    Download and save Individual household electric power consumption dataset.

    This dataset consists of almost 4 years of history with 1 minute frequency from a household in Sceaux. Different
    electrical quantities and some sub-metering values are available.

    References
    ----------
    .. [1] https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

    """
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

    dataset_dir.mkdir(exist_ok=True, parents=True)

    df = _download_dataset_zip(
        url,
        file_names="household_power_consumption.txt",
        read_functions=partial(pd.read_csv, sep=";", keep_default_na=True, na_values=["?"]),
    )

    df["timestamp"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df = df.drop(["Date", "Time"], axis=1).melt("timestamp", var_name="segment", value_name="target")
    df_full = TSDataset.to_dataset(df)

    df_full.to_csv(dataset_dir / f"IHEPC_T_full.csv.gz", index=True, compression="gzip", float_format="%.8f")


def get_australian_wine_sales_dataset(dataset_dir: Path) -> None:
    """
    Download and save Australian total wine sales by wine makers in bottles.

    This dataset consists of wine sales by Australian wine makers between Jan 1980 – Aug 1994.

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/wineind
    """
    url = (
        "https://raw.githubusercontent.com/etna-team/etna/9417d61976305ea5980e91cd06d6f33c6c7c4560/"
        "examples/data/monthly-australian-wine-sales.csv"
    )

    dataset_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(url, sep=",")
    df["timestamp"] = pd.to_datetime(df["month"])
    df["target"] = df["sales"]
    df.drop(columns=["month", "sales"], inplace=True)
    df["segment"] = "main"
    df_full = TSDataset.to_dataset(df)

    df_full.to_csv(
        dataset_dir / f"australian_wine_sales_monthly_full.csv.gz", index=True, compression="gzip", float_format="%.8f"
    )


def list_datasets() -> List[str]:
    """Return a list of available internal datasets."""
    return sorted(datasets_dict.keys())


datasets_dict: Dict[str, Dict] = {
    "electricity_15T": {
        "get_dataset_function": get_electricity_dataset_15t,
        "freq": pd.offsets.Minute(n=15),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "a3148ff2119a29f9d4c5f33bb0f7897d",
            "test": "df98e934e70e9b1dcfb0a3ee6858d76f",
            "full": "97209d3727630e6533776ce027048f71",
        },
    },
    "m3_monthly": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="monthly"),
        "freq": None,
        "exog_datetime_columns": ("origin_timestamp",),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "36535626a98157ccbfe3d1f5b2d964ac",
            "test": "09af36fa503b41ea5283db6ec6063ae1",
            "full": "4babb773e580501b4918557555157f34",
        },
    },
    "m3_quarterly": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="quarterly"),
        "freq": None,
        "exog_datetime_columns": ("origin_timestamp",),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "fb4286f519a6aa9385937c47dde6ddf4",
            "test": "a27614afc474472f842a152a6ceb95e6",
            "full": "dba2451b2aac7fc397c1cff5ad32a3dd",
        },
    },
    "m3_yearly": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="yearly"),
        "freq": None,
        "exog_datetime_columns": ("origin_timestamp",),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "1d14eb24b2dd7bc9796a5758c6b215f1",
            "test": "ad83bafa0533557a65e124aed9b1c381",
            "full": "62fc772fe16c1e0eb53401f088f82b6a",
        },
    },
    "m3_other": {
        "get_dataset_function": partial(get_m3_dataset, dataset_freq="other"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "37316d0cc7eb45c653719aea0be53880",
            "test": "a63258ce320d3f2e68c019c9f23767b1",
            "full": "81b024a7ef1b6be31e748c47edb057be",
        },
    },
    "m4_hourly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Hourly"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "239f11e69086ee0ef9c39fcb0bb89286",
            "test": "36cc4ae564342a361695c402e6812074",
            "full": "fd299eaaa9ef3deadabb0197c37ba8b2",
        },
    },
    "m4_daily": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Daily"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "7878f1485a779da34848f900c58ca991",
            "test": "e26d4a1bc0b45428a52f1ba8be3bf510",
            "full": "7a1ce18e378fb8c69f02757547ccab4c",
        },
    },
    "m4_weekly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Weekly"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "6dedd34a04fefb7f6da626b37fcf0ad2",
            "test": "69f807a621b864d7e2d51f6daca147d8",
            "full": "9954d2341af9615472f58afcc9dae2fd",
        },
    },
    "m4_monthly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Monthly"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "6c1f5212132429c24279c583d8350ec3",
            "test": "8495595ea49766f94855e2275adf41e8",
            "full": "69e4479c83174eddf22b9c125de086b8",
        },
    },
    "m4_quarterly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Quarterly"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "a82abe6bb3d471ae23dc8de0c28d62c2",
            "test": "2469cf58fea2468c30ffc4ad5891b67c",
            "full": "bc076efa89d65cb5ce35d867b9bfcb3b",
        },
    },
    "m4_yearly": {
        "get_dataset_function": partial(get_m4_dataset, dataset_freq="Yearly"),
        "freq": None,
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "b44199b886507abd9118e0f756527af9",
            "test": "676c705384b67d4ffad6d5b25873501e",
            "full": "1ee536a16c9d505f5411de5fc8e0e265",
        },
    },
    "traffic_2008_10T": {
        "get_dataset_function": partial(get_traffic_2008_dataset, dataset_freq="10T"),
        "freq": pd.offsets.Minute(n=10),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "f22f77c170e698f4f51231b24e5bc9f0",
            "test": "261ee7b09e50d1c7e1e74ccf08412f3f",
            "full": "d1d05602b15aa30d461e21148483a0c8",
        },
    },
    "traffic_2008_hourly": {
        "get_dataset_function": partial(get_traffic_2008_dataset, dataset_freq="hourly"),
        "freq": pd.offsets.Hour(),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "161748edc508b4e206344fcbb984bf9a",
            "test": "adc3fa06ee856c6481faa400e9e9f602",
            "full": "899bc1fa3fc334868a9e41033a2c3a52",
        },
    },
    "traffic_2015_hourly": {
        "get_dataset_function": get_traffic_2015_dataset,
        "freq": pd.offsets.Hour(),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "838f0b7b012cf0bf3427fb5b1a4c053f",
            "test": "67b2d13ec809f3ce58834932460793e5",
            "full": "4edf42371f28685137ac402c6a7ad2cd",
        },
    },
    "tourism_monthly": {
        "get_dataset_function": partial(get_tourism_dataset, dataset_freq="monthly"),
        "freq": None,
        "exog_datetime_columns": ("origin_timestamp",),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "eb65658979dcf20254df2e27793c4a2f",
            "test": "4413d427fb1c7fd161a2ae896a9f2e17",
            "full": "ccb8fd049488568af81c9fe341d05470",
        },
    },
    "tourism_quarterly": {
        "get_dataset_function": partial(get_tourism_dataset, dataset_freq="quarterly"),
        "freq": None,
        "exog_datetime_columns": ("origin_timestamp",),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "380fe61422a5333043b714c22bcb6725",
            "test": "0cea851864a96c344778037d3baaedf5",
            "full": "a58fd54e937182b52220c7e733b982ca",
        },
    },
    "tourism_yearly": {
        "get_dataset_function": partial(get_tourism_dataset, dataset_freq="yearly"),
        "freq": None,
        "exog_datetime_columns": ("origin_timestamp",),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "62ccbd0a636fd8797d20eab58d78e503",
            "test": "52d826295bf39cca8ab067c04e0fb883",
            "full": "33bc585db54a4b965149ff9b991c2def",
        },
    },
    "weather_10T": {
        "get_dataset_function": get_weather_dataset,
        "freq": pd.offsets.Minute(n=10),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "70ffe726161c200ae785643b38e33a11",
            "test": "a4808adbba4a50de5e4ece42ed44a333",
            "full": "cb3deebbed9b20bcd8d7c501644bf840",
        },
    },
    "ETTm1": {
        "get_dataset_function": partial(get_ett_dataset, dataset_type="ETTm1"),
        "freq": pd.offsets.Minute(n=15),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "ea71e6ca40d872916ae62d6182004a22",
            "test": "cb662ba54159a0ab505206be054be582",
            "full": "b40f1678ee1dbc764c609139120d129f",
        },
    },
    "ETTm2": {
        "get_dataset_function": partial(get_ett_dataset, dataset_type="ETTm2"),
        "freq": pd.offsets.Minute(n=15),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "e7012a0ff1847bf35050f67ddf843ce6",
            "test": "87a2409da835c27d68e5770c07b51bc3",
            "full": "d48bb6c5c4aa0deef90db9306451e1ff",
        },
    },
    "ETTh1": {
        "get_dataset_function": partial(get_ett_dataset, dataset_type="ETTh1"),
        "freq": pd.offsets.Hour(),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "c86c169fd7031c49aab23baf0e0ded5e",
            "test": "f11417d67371bc82c00ccbb044f5d1de",
            "full": "5bbf6b7045cc260893f93ef89f3346e3",
        },
    },
    "ETTh2": {
        "get_dataset_function": partial(get_ett_dataset, dataset_type="ETTh2"),
        "freq": pd.offsets.Hour(),
        "parts": ("train", "test", "full"),
        "hash": {
            "train": "58606e10507b32963a1cca89716f68a2",
            "test": "de23fa6a93c84d82f657a38958007d1c",
            "full": "11786b012971b0d97171fbc1f4e7e045",
        },
    },
    "IHEPC_T": {
        "get_dataset_function": get_ihepc_dataset,
        "freq": pd.offsets.Minute(),
        "parts": ("full",),
        "hash": {"full": "8909138462ea130b9809907e947ffae6"},
    },
    "australian_wine_sales_monthly": {
        "get_dataset_function": get_australian_wine_sales_dataset,
        "freq": pd.offsets.MonthBegin(),
        "parts": ("full",),
        "hash": {"full": "a44a58333fb79678a275c96a0160f756"},
    },
}
