from os import unlink

import numpy as np
import pandas as pd
import pytest
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from typing_extensions import NamedTuple

from etna.auto.utils import config_hash
from etna.datasets import TSDataset
from etna.models import NaiveModel
from etna.pipeline import Pipeline


@pytest.fixture()
def optuna_storage():
    yield RDBStorage("sqlite:///test.db")
    unlink("test.db")


@pytest.fixture()
def trials():
    class Trial(NamedTuple):
        user_attrs: dict
        state: TrialState = TrialState.COMPLETE

    complete_trials = [
        Trial(
            user_attrs={
                "pipeline": pipeline.to_dict(),
                "SMAPE_median": float(i),
                "hash": config_hash(pipeline.to_dict()),
            }
        )
        for i, pipeline in enumerate((Pipeline(NaiveModel(j), horizon=7) for j in range(10)))
    ]
    fail_trials = [Trial(user_attrs={}, state=TrialState.FAIL)]

    return complete_trials + complete_trials[:3] + fail_trials


@pytest.fixture
def ts_with_fold_missing_tail(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1.loc[df1.index[-7:], "target"] = np.NaN

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)
    df2.loc[df2.index[-7:], "target"] = np.NaN

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds


@pytest.fixture
def ts_with_fold_missing_middle(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1.loc[df1.index[-14:-7], "target"] = np.NaN

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)
    df2.loc[df2.index[-14:-7], "target"] = np.NaN

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds


@pytest.fixture
def ts_with_all_folds_missing_one_segment(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1.loc[df1.index[-21:], "target"] = np.NaN

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds


@pytest.fixture
def ts_with_few_missing(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1.loc[df1.index[-4:-2], "target"] = np.NaN

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)
    df2.loc[df2.index[-12:-10], "target"] = np.NaN

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds
