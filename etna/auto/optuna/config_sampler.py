from functools import partial
from typing import Optional
from typing import Set

import numpy as np
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from etna.auto.utils import retry


class ConfigSampler(BaseSampler):
    """Optuna based sampler for greedy search over various hashes of configs.

     Mapping from hashes to configs is passed directly to objective.

    The core difference with :py:class:`optuna.samplers.GridSampler` is that we want to get current trial params
    (``params`` + ``relative_params``) in objective function.
    But :py:class:`optuna.samplers.GridSampler` does not allow this, because it only implements ``sample_independent``
    which isn't suitable for this.
    """

    def __init__(
        self, config_hashes: Set[str], random_generator: Optional[np.random.Generator] = None, retries: int = 10
    ):
        """Init Config sampler.

        Parameters
        ----------
        config_hashes:
            set of config hashes to sample from
        random_generator:
            numpy generator to get reproducible samples
        retries:
            number of retries to get new sample from storage. It could be useful if storage is not reliable.
        """
        self.config_hashes = config_hashes
        self._rng = random_generator
        self.retries = retries

    def sample_independent(self, *args, **kwargs):  # noqa: D102
        """Sample independent. Not used."""
        return {}

    def infer_relative_search_space(self, *args, **kwargs):  # noqa: D102
        """Infer relative search space. Not used."""
        return {}

    def sample_relative(self, study: Study, trial: FrozenTrial, *args, **kwargs) -> dict:
        """Sample config hash to test.

        Parameters
        ----------
        study:
            current optuna study
        trial:
            optuna trial to use

        Return
        ------
        :
            sampled config hash to run objective on
        """
        trials_to_sample = self._get_unfinished_hashes(study=study, current_trial=trial)

        if len(trials_to_sample) == 0:
            # This case may occur with distributed optimization or trial queue. If there is no
            # target grid, `ConfigSampler` evaluates a visited, duplicated point with the current
            # trial. After that, the optimization stops.
            _to_sample = list(self.config_hashes)
            idx = self.rng.choice(len(_to_sample))
            hash_to_sample = _to_sample[idx]
        else:
            _trials_to_sample = list(trials_to_sample)
            idx = self.rng.choice(len(_trials_to_sample))
            hash_to_sample = _trials_to_sample[idx]

        study._storage.set_trial_user_attr(trial._trial_id, "hash", hash_to_sample)
        return {"hash": hash_to_sample}

    def after_trial(self, study: Study, trial: FrozenTrial, *args, **kwargs) -> None:  # noqa: D102
        """Stop study if all configs have been tested.

        Parameters
        ----------
        study:
            current optuna study
        """
        unfinished_hashes = self._get_unfinished_hashes(study=study, current_trial=trial)
        hash_to_sample = trial.user_attrs["hash"]

        if len(unfinished_hashes) == 0:
            study.stop()
        if len(unfinished_hashes) == 1 and unfinished_hashes.pop() == hash_to_sample:
            study.stop()

    def _get_unfinished_hashes(self, study: Study, current_trial: Optional[FrozenTrial] = None) -> Set[str]:
        """Get unfinished config hashes.

        Parameters
        ----------
        study:
            current optuna study

        Returns
        -------
        :
            hashes to run
        """
        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        if current_trial is not None:
            trials = [trial for trial in trials if trial._trial_id != current_trial._trial_id]

        finished_trials_hash = []
        running_trials_hash = []

        for t in trials:
            if t.state.is_finished():
                finished_trials_hash.append(t.user_attrs["hash"])
            elif t.state == TrialState.RUNNING:

                def _closure(trial):
                    return study._storage.get_trial(trial._trial_id).user_attrs.get("hash")

                hash_to_add = retry(partial(_closure, trial=t), max_retries=self.retries)
                # if we don't get hash, it could lead to duplication
                if hash_to_add is not None:
                    running_trials_hash.append(hash_to_add)
            else:
                pass

        unfinished_hash = set(self.config_hashes) - set(finished_trials_hash) - set(running_trials_hash)

        # If evaluations for all hashes have been started, return hashes that have not yet finished
        # because all hashes should be evaluated before stopping the optimization.
        # This logic is copied from `GridSampler`
        if len(unfinished_hash) == 0:
            unfinished_hash = set(self.config_hashes) - set(finished_trials_hash)

        return unfinished_hash

    @property
    def rng(self):  # noqa: D102
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng
