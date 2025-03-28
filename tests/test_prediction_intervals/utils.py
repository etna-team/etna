from typing import Callable
from typing import Optional

import optuna

from etna.auto.utils import suggest_parameters
from etna.datasets import TSDataset
from etna.prediction_intervals import BasePredictionIntervals


def assert_sampling_is_valid(
    intervals_pipeline: BasePredictionIntervals,
    ts: TSDataset,
    seed: int = 0,
    n_trials: int = 3,
    skip_parameters: Optional[Callable] = None,
):
    params_to_tune = intervals_pipeline.params_to_tune()

    def _objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(trial, params_to_tune)
        if skip_parameters is None or not skip_parameters(parameters):
            new_intervals_pipeline = intervals_pipeline.set_params(**parameters)
            new_intervals_pipeline.fit(ts)
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials)
