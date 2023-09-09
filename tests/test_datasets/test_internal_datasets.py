import shutil

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import load_dataset
from etna.datasets.internal_datasets import _DOWNLOAD_PATH
from etna.datasets.internal_datasets import datasets_dict


def get_custom_dataset(dataset_dir):
    np.random.seed(1)
    dataset_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(np.random.normal(0, 10, size=(30, 5)))
    df["timestamp"] = pd.date_range("2021-01-01", periods=30, freq="D")
    dt_list = df["timestamp"].values
    df = df.melt("timestamp", var_name="segment", value_name="target")
    df_train = df[df["timestamp"].isin(dt_list[:-2])]
    df_test = df[df["timestamp"].isin(dt_list[-2:])]
    TSDataset.to_dataset(df).to_csv(dataset_dir / "custom_internal_dataset_full.csv.gz", index=True, compression="gzip")
    TSDataset.to_dataset(df_train).to_csv(
        dataset_dir / "custom_internal_dataset_train.csv.gz", index=True, compression="gzip"
    )
    TSDataset.to_dataset(df_test).to_csv(
        dataset_dir / "custom_internal_dataset_test.csv.gz", index=True, compression="gzip"
    )


def update_dataset_dict(dataset_name, get_dataset_function, freq):
    datasets_dict[dataset_name] = {
        "get_dataset_function": get_dataset_function,
        "freq": freq,
        "parts": ("train", "test", "full"),
    }


def test_not_present_dataset():
    with pytest.raises(NotImplementedError, match="is not available"):
        _ = load_dataset(name="not_implemented_dataset")


def test_load_custom_dataset():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq="D")
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    ts_init = load_dataset("custom_internal_dataset", rebuild_dataset=False, parts="full")
    ts_local = load_dataset("custom_internal_dataset", rebuild_dataset=False, parts="full")
    ts_rebuild = load_dataset("custom_internal_dataset", rebuild_dataset=True, parts="full")
    shutil.rmtree(dataset_path)
    pd.util.testing.assert_frame_equal(ts_init.to_pandas(), ts_local.to_pandas())
    pd.util.testing.assert_frame_equal(ts_init.to_pandas(), ts_rebuild.to_pandas())


def test_load_all_parts():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq="D")
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    ts_train, ts_test, ts_full = load_dataset("custom_internal_dataset", parts=("train", "test", "full"))
    shutil.rmtree(dataset_path)
    assert ts_train.df.shape[0] + ts_test.df.shape[0] == ts_full.df.shape[0]


def test_not_present_part():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq="D")
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    with pytest.raises(NotImplementedError, match="is not available"):
        _ = load_dataset("custom_internal_dataset", parts="val")


@pytest.mark.skip(reason="Dataset is too large for testing in GitHub.")
@pytest.mark.parametrize(
    "dataset_name, expected_shape, expected_min_date, expected_max_date",
    [("electricity_15T", (140256, 370), pd.to_datetime("2011-01-01 00:15:00"), pd.to_datetime("2015-01-01 00:00:00"))],
)
def test_dataset_statistics(dataset_name, expected_shape, expected_min_date, expected_max_date):
    ts = load_dataset(dataset_name, parts="full")
    assert ts.df.shape == expected_shape
    assert ts.index.min() == expected_min_date
    assert ts.index.max() == expected_max_date
