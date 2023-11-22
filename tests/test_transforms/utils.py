import pathlib
import tempfile
from copy import deepcopy
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple

import optuna
import pandas as pd

from etna.auto.utils import suggest_parameters
from etna.datasets import TSDataset
from etna.transforms import Transform


def get_loaded_transform(transform: Transform) -> Transform:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        transform.save(path)
        loaded_transform = deepcopy(transform).load(path)
    return loaded_transform


def assert_transformation_equals_loaded_original(transform: Transform, ts: TSDataset) -> Tuple[Transform, Transform]:
    transform.fit(ts)
    loaded_transform = get_loaded_transform(transform)
    ts_1 = deepcopy(ts)
    ts_2 = deepcopy(ts)

    ts_1.transform([transform])
    ts_2.transform([loaded_transform])

    pd.testing.assert_frame_equal(ts_1.to_pandas(), ts_2.to_pandas())

    return transform, loaded_transform


def assert_sampling_is_valid(
    transform: Transform, ts: TSDataset, seed: int = 0, n_trials: int = 3, skip_parameters: Optional[Callable] = None
):
    params_to_tune = transform.params_to_tune()

    def _objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(trial, params_to_tune)
        if skip_parameters is None or not skip_parameters(parameters):
            new_transform = transform.set_params(**parameters)
            new_transform.fit(ts)
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials)


def find_columns_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Tuple[Set[str], Set[str], Set[str]]:
    columns_before_transform = set(df_before.columns)
    columns_after_transform = set(df_after.columns)
    created_columns = columns_after_transform - columns_before_transform
    removed_columns = columns_before_transform - columns_after_transform

    columns_to_check_changes = columns_after_transform.intersection(columns_before_transform)
    changed_columns = set()
    for column in columns_to_check_changes:
        if not df_before[column].equals(df_after[column]):
            changed_columns.add(column)

    return created_columns, removed_columns, changed_columns


def convert_ts_to_int_timestamp(ts: pd.DataFrame, bias=0):
    df = ts.to_pandas(features=["target"])
    df_exog = ts.df_exog

    ref_point = df.index[0]
    df.index = df.index.map(lambda x: (x - ref_point).days + bias)

    if df_exog is not None:
        df_exog.index = df_exog.index.map(lambda x: (x - ref_point).days + bias)

    ts = TSDataset(df=df, df_exog=df_exog, known_future=ts.known_future, freq=None)
    return ts


def assert_column_changes(ts_1: TSDataset, ts_2: TSDataset, expected_changes: Dict[str, Set[str]]):
    expected_columns_to_create = expected_changes.get("create", set())
    expected_columns_to_remove = expected_changes.get("remove", set())
    expected_columns_to_change = expected_changes.get("change", set())
    flat_df_1 = ts_1.to_pandas(flatten=True)
    flat_df_2 = ts_2.to_pandas(flatten=True)
    created_columns, removed_columns, changed_columns = find_columns_diff(flat_df_1, flat_df_2)

    assert created_columns == expected_columns_to_create
    assert removed_columns == expected_columns_to_remove
    assert changed_columns == expected_columns_to_change
