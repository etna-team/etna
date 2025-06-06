import os
import time
from random import SystemRandom

import optuna
import pytest
from joblib import Parallel
from joblib import delayed
from optuna.storages import RDBStorage

from etna.auto.optuna import ConfigSampler


@pytest.fixture()
def config_sampler():
    return ConfigSampler(config_hashes={f"hash_{i}" for i in range(10)})


@pytest.fixture()
def objective(config_mapping={f"hash_{i}": {"x": i} for i in range(10)}):
    def objective(trial: optuna.trial.Trial):
        rng = SystemRandom()
        config = config_mapping[trial.relative_params["hash"]]
        time.sleep(10 * rng.random())
        return (config["x"] - 2) ** 2

    return objective


@pytest.fixture()
def sqlite_storage():
    storage_name = f"{time.monotonic()}.db"
    yield RDBStorage(f"sqlite:///{storage_name}")
    os.unlink(storage_name)


def test_config_sampler_one_thread(
    objective, config_sampler, expected_pipeline={"x": 2}, config_mapping={f"hash_{i}": {"x": i} for i in range(10)}
):

    study = optuna.create_study(sampler=config_sampler)
    study.optimize(objective, n_trials=100)
    assert config_mapping[study.best_trial.user_attrs["hash"]] == expected_pipeline
    assert len(study.trials) == len(config_sampler.config_hashes)


def test_config_sampler_multithread_without_trials_count_check(
    objective,
    config_sampler,
    sqlite_storage,
    n_jobs=4,
    expected_pipeline={"x": 2},
    config_mapping={f"hash_{i}": {"x": i} for i in range(10)},
):

    study = optuna.create_study(sampler=config_sampler, storage=sqlite_storage)
    Parallel(n_jobs=n_jobs)(delayed(study.optimize)(objective) for _ in range(n_jobs))

    assert config_mapping[study.best_trial.user_attrs["hash"]] == expected_pipeline


@pytest.mark.skip(reason="The number of trials is non-deterministic")
def test_config_sampler_multithread(objective, config_sampler, sqlite_storage, n_jobs=4, expected_pipeline={"x": 2}):

    study = optuna.create_study(sampler=config_sampler, storage=sqlite_storage)
    Parallel(n_jobs=n_jobs)(delayed(study.optimize)(objective) for _ in range(n_jobs))

    assert study.best_trial.user_attrs["pipeline"] == expected_pipeline
    # TODO: this test case is non-deterministic
    assert len(study.trials) == len(config_sampler.config_hashes) + n_jobs - 1
