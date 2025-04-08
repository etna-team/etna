import numpy as np
import pandas as pd
import pytest

from etna.datasets import generate_ar_df
from etna.datasets.hierarchical_structure import HierarchicalStructure


@pytest.fixture
def long_hierarchical_structure():
    hs = HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a"], "a": ["c"], "Y": ["b"], "b": ["d"]},
        level_names=["l1", "l2", "l3", "l4"],
    )
    return hs


@pytest.fixture
def tailed_hierarchical_structure():
    hs = HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"], "c": ["f"], "d": ["g"], "a": ["e", "h"]},
        level_names=["l1", "l2", "l3", "l4"],
    )
    return hs


@pytest.fixture
def df_aligned_datetime() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq=pd.offsets.Day())
    return df


@pytest.fixture
def df_exog_aligned_datetime() -> pd.DataFrame:
    df_exog_1 = generate_ar_df(start_time="2020-01-01", periods=15, n_segments=2, freq=pd.offsets.Day())
    df_exog_1.rename(columns={"target": "exog_1"}, inplace=True)

    df_exog_2 = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq=pd.offsets.Day())
    df_exog_2.rename(columns={"target": "exog_2"}, inplace=True)

    df = pd.merge(left=df_exog_1, right=df_exog_2, how="outer")

    return df


@pytest.fixture
def df_aligned_int() -> pd.DataFrame:
    df = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    return df


@pytest.fixture
def df_exog_aligned_int() -> pd.DataFrame:
    df_exog_1 = generate_ar_df(start_time=10, periods=15, n_segments=2, freq=None)
    df_exog_1.rename(columns={"target": "exog_1"}, inplace=True)

    df_exog_2 = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    df_exog_2.rename(columns={"target": "exog_2"}, inplace=True)

    df = pd.merge(left=df_exog_1, right=df_exog_2, how="outer")

    return df


@pytest.fixture
def df_misaligned_datetime() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq=pd.offsets.Day())
    df = df.iloc[:-3]
    return df


@pytest.fixture
def df_exog_misaligned_datetime() -> pd.DataFrame:
    df_exog_1 = generate_ar_df(start_time="2020-01-01", periods=15, n_segments=2, freq=pd.offsets.Day())
    df_exog_1.rename(columns={"target": "exog_1"}, inplace=True)
    df_exog_1 = df_exog_1.iloc[:-3]

    df_exog_2 = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq=pd.offsets.Day())
    df_exog_2.rename(columns={"target": "exog_2"}, inplace=True)
    df_exog_2 = df_exog_2.iloc[:-3]

    df = pd.merge(left=df_exog_1, right=df_exog_2, how="outer")

    return df


@pytest.fixture
def df_exog_all_misaligned_datetime() -> pd.DataFrame:
    df_exog_1 = generate_ar_df(start_time="2020-01-01", periods=20, n_segments=2, freq=pd.offsets.Day())
    df_exog_1.rename(columns={"target": "exog_1"}, inplace=True)
    df_exog_1 = df_exog_1.iloc[:-3]

    df_exog_2 = generate_ar_df(start_time="2020-01-01", periods=20, n_segments=2, freq=pd.offsets.Day())
    df_exog_2.rename(columns={"target": "exog_2"}, inplace=True)
    df_exog_2 = df_exog_2.iloc[:-3]

    df = pd.merge(left=df_exog_1, right=df_exog_2, how="outer")

    return df


@pytest.fixture
def df_misaligned_int() -> pd.DataFrame:
    df = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    df = df.iloc[:-3]
    return df


@pytest.fixture
def df_exog_misaligned_int() -> pd.DataFrame:
    df_exog_1 = generate_ar_df(start_time=10, periods=15, n_segments=2, freq=None)
    df_exog_1.rename(columns={"target": "exog_1"}, inplace=True)
    df_exog_1 = df_exog_1.iloc[:-3]

    df_exog_2 = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    df_exog_2.rename(columns={"target": "exog_2"}, inplace=True)
    df_exog_2 = df_exog_2.iloc[:-3]

    df = pd.merge(left=df_exog_1, right=df_exog_2, how="outer")

    return df


@pytest.fixture
def df_aligned_datetime_with_missing_values() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq=pd.offsets.Day())
    df.loc[df.index[-3:], "target"] = np.NaN
    return df


@pytest.fixture
def df_aligned_int_with_missing_values() -> pd.DataFrame:
    df = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    df.loc[df.index[-3:], "target"] = np.NaN
    return df


@pytest.fixture
def df_aligned_datetime_with_additional_columns() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq=pd.offsets.Day())
    df["feature_1"] = df["timestamp"].dt.weekday
    return df
