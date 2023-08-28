import shutil

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets.internal_datasets import load_dataset
from etna.datasets.internal_datasets import _DOWNLOAD_PATH
from etna.datasets.internal_datasets import datasets_dict


def get_custom_dataset(dataset_dir):
    np.random.seed(1)
    dataset_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(np.random.normal(0, 10, size=(30, 5)))
    df["timestamp"] = pd.date_range("2021-01-01", periods=30, freq="D")
    df = df.melt("timestamp", var_name="segment", value_name="target")
    df.to_csv(dataset_dir / "custom_internal_dataset.csv", index=False)
    ts = TSDataset(TSDataset.to_dataset(df), freq="D")
    return ts


def update_dataset_dict(dataset_name, get_dataset_function, freq):
    datasets_dict[dataset_name] = {"get_dataset_function": get_dataset_function, "freq": freq}


def test_not_present_dataset():
    with pytest.raises(NotImplementedError, match="is not available."):
        _ = load_dataset(name="not_implemented_dataset")


def test_load_custom_dataset():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq="D")
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    ts_init = load_dataset("custom_internal_dataset", rebuild_dataset=False)
    ts_local = load_dataset("custom_internal_dataset", rebuild_dataset=False)
    ts_rebuild = load_dataset("custom_internal_dataset", rebuild_dataset=True)
    shutil.rmtree(dataset_path)
    pd.util.testing.assert_frame_equal(ts_init.to_pandas(), ts_local.to_pandas())
    pd.util.testing.assert_frame_equal(ts_init.to_pandas(), ts_rebuild.to_pandas())


@pytest.mark.parametrize(
    "dataset_name, expected_shape, expected_min_date, expected_max_date",
    [("electricity", (140256, 370), pd.to_datetime("2011-01-01 00:15:00"), pd.to_datetime("2015-01-01 00:00:00"))],
)
def test_dataset_statistics(dataset_name, expected_shape, expected_min_date, expected_max_date):
    ts = load_dataset(dataset_name)
    assert ts.df.shape == expected_shape
    assert ts.index.min() == expected_min_date
    assert ts.index.max() == expected_max_date
