import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import load_dataset
from etna.datasets.internal_datasets import _DOWNLOAD_PATH
from etna.datasets.internal_datasets import datasets_dict
from etna.datasets.internal_datasets import list_datasets
from etna.datasets.internal_datasets import read_dataset


def get_custom_dataset(dataset_dir):
    np.random.seed(1)
    dataset_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(np.random.normal(0, 10, size=(30, 5)))
    df["timestamp"] = pd.date_range("2021-01-01", periods=30, freq=pd.offsets.Day())
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
        "hash": {"train": "", "test": "", "full": ""},
    }


def test_not_present_dataset():
    with pytest.raises(NotImplementedError, match="is not available"):
        _ = load_dataset(name="not_implemented_dataset")


@pytest.mark.filterwarnings("ignore: Local hash and expected hash are different for")
def test_load_custom_dataset():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq=pd.offsets.Day())
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    ts_init = load_dataset("custom_internal_dataset", rebuild_dataset=False, parts="full")
    ts_local = load_dataset("custom_internal_dataset", rebuild_dataset=False, parts="full")
    ts_rebuild = load_dataset("custom_internal_dataset", rebuild_dataset=True, parts="full")
    shutil.rmtree(dataset_path)
    pd.testing.assert_frame_equal(ts_init.to_pandas(), ts_local.to_pandas())
    pd.testing.assert_frame_equal(ts_init.to_pandas(), ts_rebuild.to_pandas())


@pytest.mark.filterwarnings("ignore: Local hash and expected hash are different for")
def test_load_all_parts():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq=pd.offsets.Day())
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    ts_train, ts_test, ts_full = load_dataset("custom_internal_dataset", parts=("train", "test", "full"))
    shutil.rmtree(dataset_path)
    assert ts_train.size()[0] + ts_test.size()[0] == ts_full.size()[0]


def test_not_present_part():
    update_dataset_dict(dataset_name="custom_internal_dataset", get_dataset_function=get_custom_dataset, freq=pd.offsets.Day())
    dataset_path = _DOWNLOAD_PATH / "custom_internal_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    with pytest.raises(NotImplementedError, match="is not available"):
        _ = load_dataset("custom_internal_dataset", parts="val")


@pytest.mark.parametrize(
    "dataset_name, expected_shape, expected_min_timestamp, expected_max_timestamp, dataset_parts",
    [
        pytest.param(
            "electricity_15T",
            (139896 + 360, 370),
            pd.to_datetime("2011-01-01 00:15:00"),
            pd.to_datetime("2015-01-01 00:00:00"),
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        (
            "m4_hourly",
            (960 + 48, 414),
            0,
            1007,
            ("train", "test"),
        ),
        pytest.param(
            "m4_daily",
            (9919 + 14, 4227),
            0,
            9932,
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        (
            "m4_weekly",
            (2597 + 13, 359),
            0,
            2609,
            ("train", "test"),
        ),
        pytest.param(
            "m4_monthly",
            (2794 + 18, 48000),
            0,
            2811,
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        (
            "m4_quarterly",
            (866 + 8, 24000),
            0,
            873,
            ("train", "test"),
        ),
        (
            "m4_yearly",
            (835 + 6, 23000),
            0,
            840,
            ("train", "test"),
        ),
        pytest.param(
            "traffic_2008_10T",
            (65376 + 144, 963),
            pd.to_datetime("2008-01-01 00:00:00"),
            pd.to_datetime("2009-03-30 23:50:00"),
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        pytest.param(
            "traffic_2008_hourly",
            (10896 + 24, 963),
            pd.to_datetime("2008-01-01 00:00:00"),
            pd.to_datetime("2009-03-30 23:00:00"),
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        pytest.param(
            "traffic_2015_hourly",
            (17520 + 24, 862),
            pd.to_datetime("2015-01-01 00:00:00"),
            pd.to_datetime("2016-12-31 23:00:00"),
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        (
            "m3_monthly",
            (126 + 18, 2856),
            0,
            143,
            ("train", "test"),
        ),
        (
            "m3_quarterly",
            (64 + 8, 1512),
            0,
            71,
            ("train", "test"),
        ),
        (
            "m3_other",
            (96 + 8, 348),
            0,
            103,
            ("train", "test"),
        ),
        (
            "m3_yearly",
            (41 + 6, 1290),
            0,
            46,
            ("train", "test"),
        ),
        (
            "tourism_monthly",
            (309 + 24, 732),
            0,
            332,
            ("train", "test"),
        ),
        (
            "tourism_quarterly",
            (122 + 8, 854),
            0,
            129,
            ("train", "test"),
        ),
        (
            "tourism_yearly",
            (43 + 4, 1036),
            0,
            46,
            ("train", "test"),
        ),
        pytest.param(
            "weather_10T",
            (52560 + 144, 21),
            pd.to_datetime("2020-01-01 00:10:00"),
            pd.to_datetime("2021-01-01 00:00:00"),
            ("train", "test"),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        (
            "ETTm1",
            (66800 + 2880, 7),
            pd.to_datetime("2016-07-01 00:00:00"),
            pd.to_datetime("2018-06-26 19:45:00"),
            ("train", "test"),
        ),
        (
            "ETTm2",
            (66800 + 2880, 7),
            pd.to_datetime("2016-07-01 00:00:00"),
            pd.to_datetime("2018-06-26 19:45:00"),
            ("train", "test"),
        ),
        (
            "ETTh1",
            (16700 + 720, 7),
            pd.to_datetime("2016-07-01 00:00:00"),
            pd.to_datetime("2018-06-26 19:00:00"),
            ("train", "test"),
        ),
        (
            "ETTh2",
            (16700 + 720, 7),
            pd.to_datetime("2016-07-01 00:00:00"),
            pd.to_datetime("2018-06-26 19:00:00"),
            ("train", "test"),
        ),
        pytest.param(
            "IHEPC_T",
            (2075259, 7),
            pd.to_datetime("2006-12-16 17:24:00"),
            pd.to_datetime("2010-11-26 21:02:00"),
            tuple(),
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        (
            "australian_wine_sales_monthly",
            (176, 1),
            pd.to_datetime("1980-01-01 00:00:00"),
            pd.to_datetime("1994-08-01 00:00:00"),
            tuple(),
        ),
    ],
)
def test_dataset_statistics(
    dataset_name, expected_shape, expected_min_timestamp, expected_max_timestamp, dataset_parts
):
    ts_full = load_dataset(dataset_name, parts="full", rebuild_dataset=True)
    assert (ts_full.size()[0], ts_full.size()[1] * ts_full.size()[2]) == expected_shape
    assert ts_full.timestamps.min() == expected_min_timestamp
    assert ts_full.timestamps.max() == expected_max_timestamp

    if dataset_parts:
        ts_parts = load_dataset(dataset_name, parts=dataset_parts)
        parts_rows = sum([ts.size()[0] for ts in ts_parts])
        assert ts_full.size()[0] == parts_rows


@pytest.mark.parametrize(
    "dataset_name, expected_df_exog_shapes, expected_df_exog_timestamps, dataset_parts",
    [
        (
            "m3_monthly",
            ((144, 1428), (144, 1428), (18, 1428)),
            (
                (0, 143),
                (0, 143),
                (126, 143),
            ),
            ("full", "train", "test"),
        ),
        (
            "m3_quarterly",
            ((72, 756), (72, 756), (8, 756)),
            (
                (0, 71),
                (0, 71),
                (64, 71),
            ),
            ("full", "train", "test"),
        ),
        (
            "m3_yearly",
            ((47, 645), (47, 645), (6, 645)),
            (
                (0, 46),
                (0, 46),
                (41, 46),
            ),
            ("full", "train", "test"),
        ),
        (
            "tourism_monthly",
            ((333, 366), (333, 366), (24, 366)),
            (
                (0, 332),
                (0, 332),
                (309, 332),
            ),
            ("full", "train", "test"),
        ),
        (
            "tourism_quarterly",
            ((130, 427), (130, 427), (8, 427)),
            (
                (0, 129),
                (0, 129),
                (122, 129),
            ),
            ("full", "train", "test"),
        ),
        (
            "tourism_yearly",
            ((47, 518), (47, 518), (4, 518)),
            (
                (0, 46),
                (0, 46),
                (43, 46),
            ),
            ("full", "train", "test"),
        ),
    ],
)
def test_df_exog_statistics(
    dataset_name,
    expected_df_exog_shapes,
    expected_df_exog_timestamps,
    dataset_parts,
):
    ts_parts = load_dataset(dataset_name, parts=dataset_parts)
    for i, part in enumerate(ts_parts):
        assert part._df_exog.shape == expected_df_exog_shapes[i]
    for i, part in enumerate(ts_parts):
        assert (part._df_exog.index.min(), part._df_exog.index.max()) == expected_df_exog_timestamps[i]
    for i, part in enumerate(ts_parts):
        exog_col_type = part._df_exog.dtypes.iloc[0]
        assert pd.api.types.is_datetime64_dtype(exog_col_type)


def test_list_datasets():
    datasets_names = list_datasets()
    assert len(datasets_names) == len(datasets_dict)


@pytest.mark.parametrize(
    "dataset_name",
    [
        pytest.param(
            "electricity_15T",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        "m4_hourly",
        pytest.param(
            "m4_daily",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        "m4_weekly",
        pytest.param(
            "m4_monthly",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        "m4_quarterly",
        "m4_yearly",
        pytest.param(
            "traffic_2008_10T",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        pytest.param(
            "traffic_2008_hourly",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        pytest.param(
            "traffic_2015_hourly",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        "m3_monthly",
        "m3_quarterly",
        "m3_other",
        "m3_yearly",
        "tourism_monthly",
        "tourism_quarterly",
        "tourism_yearly",
        pytest.param(
            "weather_10T",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        "ETTm1",
        "ETTm2",
        "ETTh1",
        "ETTh2",
        pytest.param(
            "IHEPC_T",
            marks=pytest.mark.skipif(
                json.loads(os.getenv("skip_large_tests", "false")), reason="Dataset is too large for testing in GitHub."
            ),
        ),
        "australian_wine_sales_monthly",
    ],
)
def test_dataset_hash(dataset_name):
    dataset_dir = _DOWNLOAD_PATH / dataset_name
    get_dataset_function = datasets_dict[dataset_name]["get_dataset_function"]
    get_dataset_function(dataset_dir)
    for part in datasets_dict[dataset_name]["hash"]:
        data, dataset_hash = read_dataset(dataset_path=dataset_dir / f"{dataset_name}_{part}.csv.gz")
        assert dataset_hash == datasets_dict[dataset_name]["hash"][part]
