import shutil

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import load_dataset
from etna.datasets.internal_datasets import _DOWNLOAD_PATH
from etna.datasets.internal_datasets import datasets_dict
from etna.datasets.internal_datasets import list_datasets


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


@pytest.mark.parametrize(
    "dataset_name, expected_shape, expected_min_date, expected_max_date, dataset_parts",
    [
        pytest.param(
            "electricity_15T",
            (139896 + 360, 370),
            pd.to_datetime("2011-01-01 00:15:00"),
            pd.to_datetime("2015-01-01 00:00:00"),
            ("train", "test"),
            marks=pytest.mark.skip(reason="Dataset is too large for testing in GitHub."),
        ),
        (
            "m4_hourly",
            (960 + 48, 414),
            pd.to_datetime("2021-11-20 01:00:00"),
            pd.to_datetime("2022-01-01 00:00:00"),
            ("train", "test"),
        ),
        (
            "m4_daily",
            (9919 + 14, 4227),
            pd.to_datetime("1994-10-23 00:00:00"),
            pd.to_datetime("2022-01-01 00:00:00"),
            ("train", "test"),
        ),
        (
            "m4_weekly",
            (2597 + 13, 359),
            pd.to_datetime("1971-12-27 00:00:00"),
            pd.to_datetime("2021-12-27 00:00:00"),
            ("train", "test"),
        ),
        (
            "m4_monthly",
            (2794 + 18, 48000),
            pd.to_datetime("1787-09-30 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "m4_quarterly",
            (866 + 8, 24000),
            pd.to_datetime("1803-10-01 00:00:00"),
            pd.to_datetime("2022-01-01 00:00:00"),
            ("train", "test"),
        ),
        (
            "m4_yearly",
            (835 + 6, 23000),
            pd.to_datetime("2019-09-14 00:00:00"),
            pd.to_datetime("2022-01-01 00:00:00"),
            ("train", "test"),
        ),
        pytest.param(
            "traffic_2008_10T",
            (65376 + 144, 963),
            pd.to_datetime("2008-01-01 00:00:00"),
            pd.to_datetime("2009-03-30 23:50:00"),
            ("train", "test"),
            marks=pytest.mark.skip(reason="Dataset is too large for testing in GitHub."),
        ),
        pytest.param(
            "traffic_2008_hourly",
            (10896 + 24, 963),
            pd.to_datetime("2008-01-01 00:00:00"),
            pd.to_datetime("2009-03-30 23:00:00"),
            ("train", "test"),
            marks=pytest.mark.skip(reason="Dataset is too large for testing in GitHub."),
        ),
        pytest.param(
            "traffic_2015_hourly",
            (17520 + 24, 862),
            pd.to_datetime("2015-01-01 00:00:00"),
            pd.to_datetime("2016-12-31 23:00:00"),
            ("train", "test"),
            marks=pytest.mark.skip(reason="Dataset is too large for testing in GitHub."),
        ),
        (
            "m3_monthly",
            (126 + 18, 2856),
            pd.to_datetime("2010-01-31 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "m3_quarterly",
            (64 + 8, 1512),
            pd.to_datetime("2004-03-31 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "m3_other",
            (96 + 8, 348),
            pd.to_datetime("1996-03-31 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "m3_yearly",
            (41 + 6, 1290),
            pd.to_datetime("1975-12-31 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "tourism_monthly",
            (309 + 24, 732),
            pd.to_datetime("1994-05-01 00:00:00"),
            pd.to_datetime("2022-01-01 00:00:00"),
            ("train", "test"),
        ),
        (
            "tourism_quarterly",
            (122 + 8, 854),
            pd.to_datetime("1989-09-30 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "tourism_yearly",
            (43 + 4, 1036),
            pd.to_datetime("1975-12-31 00:00:00"),
            pd.to_datetime("2021-12-31 00:00:00"),
            ("train", "test"),
        ),
        (
            "weather_10T",
            (52560 + 144, 21),
            pd.to_datetime("2020-01-01 00:10:00"),
            pd.to_datetime("2021-01-01 00:00:00"),
            ("train", "test"),
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
    ],
)
def test_dataset_statistics(dataset_name, expected_shape, expected_min_date, expected_max_date, dataset_parts):
    ts_full = load_dataset(dataset_name, parts="full", rebuild_dataset=True)
    ts_parts = load_dataset(dataset_name, parts=dataset_parts)
    parts_rows = sum([ts.df.shape[0] for ts in ts_parts])
    assert ts_full.df.shape == expected_shape
    assert ts_full.index.min() == expected_min_date
    assert ts_full.index.max() == expected_max_date
    assert ts_full.df.shape[0] == parts_rows


@pytest.mark.parametrize(
    "dataset_name, expected_df_exog_shapes, expected_df_exog_dates, dataset_parts",
    [
        (
            "m3_monthly",
            ((144, 1428), (144, 1428), (18, 1428)),
            (
                (pd.to_datetime("2010-01-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2010-01-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2020-07-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
        (
            "m3_quarterly",
            ((72, 756), (72, 756), (8, 756)),
            (
                (pd.to_datetime("2004-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2004-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2020-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
        (
            "m3_other",
            ((104, 174), (104, 174), (8, 174)),
            (
                (pd.to_datetime("1996-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("1996-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2020-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
        (
            "m3_yearly",
            ((47, 645), (47, 645), (6, 645)),
            (
                (pd.to_datetime("1975-12-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("1975-12-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2016-12-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
        (
            "tourism_monthly",
            ((333, 366), (333, 366), (24, 366)),
            (
                (pd.to_datetime("1994-05-01 00:00:00"), pd.to_datetime("2022-01-01 00:00:00")),
                (pd.to_datetime("1994-05-01 00:00:00"), pd.to_datetime("2022-01-01 00:00:00")),
                (pd.to_datetime("2020-02-01 00:00:00"), pd.to_datetime("2022-01-01 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
        (
            "tourism_quarterly",
            ((130, 427), (130, 427), (8, 427)),
            (
                (pd.to_datetime("1989-09-30 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("1989-09-30 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2020-03-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
        (
            "tourism_yearly",
            ((47, 518), (47, 518), (4, 518)),
            (
                (pd.to_datetime("1975-12-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("1975-12-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
                (pd.to_datetime("2018-12-31 00:00:00"), pd.to_datetime("2021-12-31 00:00:00")),
            ),
            ("full", "train", "test"),
        ),
    ],
)
def test_df_exog_statistics(
    dataset_name,
    expected_df_exog_shapes,
    expected_df_exog_dates,
    dataset_parts,
):
    ts_parts = load_dataset(dataset_name, parts=dataset_parts)
    for i, part in enumerate(ts_parts):
        assert part.df_exog.shape == expected_df_exog_shapes[i]
    for i, part in enumerate(ts_parts):
        assert (part.df_exog.index.min(), part.df_exog.index.max()) == expected_df_exog_dates[i]


def test_list_datasets():
    datasets_names = list_datasets()
    assert len(datasets_names) > 0