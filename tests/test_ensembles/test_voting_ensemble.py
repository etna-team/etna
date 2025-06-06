from copy import deepcopy
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pytest
from joblib import Parallel
from joblib import delayed
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.ensembles.voting_ensemble import VotingEnsemble
from etna.metrics import MAE
from etna.models import CatBoostPerSegmentModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import StandardScalerTransform
from tests.test_ensembles.utils import check_backtest_return_type
from tests.test_pipeline.utils import assert_pipeline_equals_loaded_original
from tests.test_pipeline.utils import assert_pipeline_forecast_raise_error_if_no_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts_with_prediction_intervals
from tests.test_pipeline.utils import assert_pipeline_forecasts_without_self_ts
from tests.test_pipeline.utils import assert_pipeline_predicts

HORIZON = 7


@pytest.mark.parametrize(
    "weights",
    (None, [0.2, 0.3, 0.5], "auto"),
)
def test_validate_weights_pass(
    weights: Optional[Union[List[float], Literal["auto"]]],
):
    """Check that VotingEnsemble._validate_weights validate weights correctly in case of valid args sets."""
    VotingEnsemble._validate_weights(weights=weights, pipelines_number=3)


def test_validate_weights_fail():
    """Check that VotingEnsemble._validate_weights validate weights correctly in case of invalid args sets."""
    with pytest.raises(ValueError, match="Weights size should be equal to pipelines number."):
        _ = VotingEnsemble._validate_weights(weights=[0.3, 0.4, 0.3], pipelines_number=2)


@pytest.mark.parametrize(
    "weights,pipelines_number,expected",
    ((None, 5, [0.2, 0.2, 0.2, 0.2, 0.2]), ([0.2, 0.3, 0.5], 3, [0.2, 0.3, 0.5]), ([1, 1, 2], 3, [0.25, 0.25, 0.5])),
)
def test_process_weights(
    example_tsdf: TSDataset,
    naive_pipeline_1: Pipeline,
    weights: Optional[List[float]],
    pipelines_number: int,
    expected: List[float],
):
    """Check that _process_weights processes weights correctly."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1 for _ in range(pipelines_number)], weights=weights)
    result = ensemble._process_weights(ts=example_tsdf)
    assert isinstance(result, list)
    assert result == expected


def test_process_weights_auto(example_tsdf: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that _process_weights processes weights correctly in "auto" mode."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights="auto")
    ensemble.ts = example_tsdf
    result = ensemble._process_weights(ts=example_tsdf)
    assert isinstance(result, list)
    assert result[0] > result[1]


@pytest.mark.parametrize(
    "weights",
    ((None, [0.2, 0.3], "auto")),
)
def test_fit_interface(
    example_tsdf: TSDataset,
    weights: Optional[Union[List[float], Literal["auto"]]],
    naive_pipeline_1: Pipeline,
    naive_pipeline_2: Pipeline,
):
    """Check that fit saves processes weights."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=weights)
    ensemble.fit(example_tsdf)
    result = ensemble.processed_weights
    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.parametrize("save_ts", [False, True])
def test_fit_saving_ts(example_tsds, naive_pipeline_1, naive_pipeline_2, save_ts):
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(example_tsds, save_ts=save_ts)

    if save_ts:
        assert ensemble.ts is example_tsds
    else:
        assert ensemble.ts is None


def test_forecast_interface(example_tsds: TSDataset, catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that VotingEnsemble.forecast returns TSDataset of correct length."""
    ensemble = VotingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline])
    ensemble.fit(ts=example_tsds)
    forecast = ensemble.forecast()
    assert isinstance(forecast, TSDataset)
    assert forecast.size()[0] == HORIZON


def test_forecast_prediction_interval_interface(example_tsds, naive_pipeline_1, naive_pipeline_2):
    """Test the forecast interface with prediction intervals."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ensemble.fit(example_tsds)
    forecast = ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_vote_default_weights(simple_tsdf, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that VotingEnsemble gets average during vote."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=simple_tsdf)
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._forecast_pipeline)(pipeline=pipeline, ts=simple_tsdf) for pipeline in ensemble.pipelines
    )
    forecast = ensemble._vote(forecasts=forecasts)
    np.testing.assert_array_equal(forecast[:, "A", "target"].values, [47.5, 48, 47.5, 48, 47.5, 48, 47.5])
    np.testing.assert_array_equal(forecast[:, "B", "target"].values, [11, 12, 11, 12, 11, 12, 11])


def test_vote_custom_weights(simple_tsdf, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that VotingEnsemble gets average during vote."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ensemble.fit(ts=simple_tsdf)
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._forecast_pipeline)(pipeline=pipeline, ts=simple_tsdf) for pipeline in ensemble.pipelines
    )
    forecast = ensemble._vote(forecasts=forecasts)
    np.testing.assert_array_equal(forecast[:, "A", "target"].values, [47.25, 48, 47.25, 48, 47.25, 48, 47.25])
    np.testing.assert_array_equal(forecast[:, "B", "target"].values, [10.5, 12, 10.5, 12, 10.5, 12, 10.5])


def test_forecast_calls_vote(example_tsds: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=example_tsds)
    ensemble._vote = MagicMock()

    result = ensemble._forecast(ts=example_tsds, return_components=False)

    ensemble._vote.assert_called_once()
    assert result == ensemble._vote.return_value


def test_predict_calls_vote(example_tsds: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=example_tsds)
    ensemble._vote = MagicMock()

    result = ensemble._predict(
        ts=example_tsds,
        start_timestamp=example_tsds.timestamps[20],
        end_timestamp=example_tsds.timestamps[30],
        prediction_interval=False,
        quantiles=(),
        return_components=False,
    )

    ensemble._vote.assert_called_once()
    assert result == ensemble._vote.return_value


def test_multiprocessing_ensembles(
    simple_tsdf,
    catboost_pipeline: Pipeline,
    prophet_pipeline: Pipeline,
    naive_pipeline_1: Pipeline,
    naive_pipeline_2: Pipeline,
):
    """Check that VotingEnsemble works the same in case of multi and single jobs modes."""
    pipelines = [catboost_pipeline, prophet_pipeline, naive_pipeline_1, naive_pipeline_2]
    single_jobs_ensemble = VotingEnsemble(pipelines=deepcopy(pipelines), n_jobs=1)
    multi_jobs_ensemble = VotingEnsemble(pipelines=deepcopy(pipelines), n_jobs=3)

    single_jobs_ensemble.fit(ts=deepcopy(simple_tsdf))
    multi_jobs_ensemble.fit(ts=deepcopy(simple_tsdf))

    single_jobs_forecast = single_jobs_ensemble.forecast()
    multi_jobs_forecast = multi_jobs_ensemble.forecast()

    assert (single_jobs_forecast._df == multi_jobs_forecast._df).all().all()


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(voting_ensemble_pipeline: VotingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that backtest works with VotingEnsemble."""
    results = voting_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    check_backtest_return_type(results, VotingEnsemble)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest_hierarchical_pipeline(
    voting_ensemble_hierarchical_pipeline: VotingEnsemble,
    product_level_simple_hierarchical_ts_long_history: TSDataset,
    n_jobs: int,
):
    """Check that backtest works with VotingEnsemble of hierarchical pipelines."""
    results = voting_ensemble_hierarchical_pipeline.backtest(
        ts=product_level_simple_hierarchical_ts_long_history, metrics=[MAE()], n_jobs=n_jobs, n_folds=3
    )
    check_backtest_return_type(results, VotingEnsemble)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest_mix_pipeline(
    voting_ensemble_mix_pipeline: VotingEnsemble,
    product_level_simple_hierarchical_ts_long_history: TSDataset,
    n_jobs: int,
):
    """Check that backtest works with VotingEnsemble of pipeline and hierarchical pipeline."""
    results = voting_ensemble_mix_pipeline.backtest(
        ts=product_level_simple_hierarchical_ts_long_history, metrics=[MAE()], n_jobs=n_jobs, n_folds=3
    )
    check_backtest_return_type(results, VotingEnsemble)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_get_historical_forecasts(voting_ensemble_pipeline: VotingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that get_historical_forecasts works with VotingEnsemble."""
    n_folds = 3
    forecast_ts_list = voting_ensemble_pipeline.get_historical_forecasts(
        ts=example_tsds, n_jobs=n_jobs, n_folds=n_folds
    )
    assert isinstance(forecast_ts_list, List)
    for forecast_ts in forecast_ts_list:
        assert isinstance(forecast_ts, TSDataset)
        assert forecast_ts.size()[0] == HORIZON


@pytest.mark.parametrize("load_ts", [True, False])
def test_save_load(load_ts, voting_ensemble_pipeline, example_tsds):
    assert_pipeline_equals_loaded_original(pipeline=voting_ensemble_pipeline, ts=example_tsds, load_ts=load_ts)


def test_forecast_raise_error_if_no_ts(voting_ensemble_pipeline, example_tsds):
    assert_pipeline_forecast_raise_error_if_no_ts(pipeline=voting_ensemble_pipeline, ts=example_tsds)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "voting_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "voting_ensemble_pipeline_int_timestamp"),
    ],
)
def test_forecasts_without_self_ts(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_without_self_ts(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "voting_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "voting_ensemble_pipeline_int_timestamp"),
    ],
)
def test_forecast_given_ts(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_given_ts(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "voting_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "voting_ensemble_pipeline_int_timestamp"),
    ],
)
def test_forecast_given_ts_with_prediction_interval(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_given_ts_with_prediction_intervals(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "voting_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "voting_ensemble_pipeline_int_timestamp"),
    ],
)
def test_predict(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_predicts(pipeline=ensemble, ts=ts, start_idx=20, end_idx=30)


def test_forecast_with_return_components_fails(example_tsds, voting_ensemble_naive):
    voting_ensemble_naive.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="Adding target components is not currently implemented!"):
        voting_ensemble_naive.forecast(return_components=True)


def test_predict_with_return_components_fails(example_tsds, voting_ensemble_naive):
    voting_ensemble_naive.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="Adding target components is not currently implemented!"):
        voting_ensemble_naive.predict(ts=example_tsds, return_components=True)


@pytest.mark.parametrize(
    "pipeline_0_tune_params, pipeline_1_tune_params, expected_tune_params",
    [
        (
            {
                "model.alpha": [0, 3, 5],
                "model.beta": [0.1, 0.2, 0.3],
                "transforms.0.param_1": ["option_1", "option_2"],
                "transforms.0.param_2": [False, True],
                "transforms.1.param_1": [1, 2],
            },
            {
                "model.alpha": [0, 3, 5],
                "model.beta": [0.1, 0.2, 0.3],
                "transforms.0.param_1": ["option_1", "option_2"],
                "transforms.0.param_2": [False, True],
                "transforms.1.param_1": [1, 2],
            },
            {
                "pipelines.0.model.alpha": [0, 3, 5],
                "pipelines.0.model.beta": [0.1, 0.2, 0.3],
                "pipelines.0.transforms.0.param_1": ["option_1", "option_2"],
                "pipelines.0.transforms.0.param_2": [False, True],
                "pipelines.0.transforms.1.param_1": [1, 2],
                "pipelines.1.model.alpha": [0, 3, 5],
                "pipelines.1.model.beta": [0.1, 0.2, 0.3],
                "pipelines.1.transforms.0.param_1": ["option_1", "option_2"],
                "pipelines.1.transforms.0.param_2": [False, True],
                "pipelines.1.transforms.1.param_1": [1, 2],
            },
        )
    ],
)
def test_params_to_tune_mocked(pipeline_0_tune_params, pipeline_1_tune_params, expected_tune_params):
    pipeline_0 = MagicMock()
    pipeline_0.params_to_tune.return_value = pipeline_0_tune_params
    pipeline_0.horizon = 1

    pipeline_1 = MagicMock()
    pipeline_1.params_to_tune.return_value = pipeline_1_tune_params
    pipeline_1.horizon = 1

    voting_ensemble_pipeline = VotingEnsemble(pipelines=[pipeline_0, pipeline_1])

    assert voting_ensemble_pipeline.params_to_tune() == expected_tune_params


@pytest.mark.parametrize(
    "pipelines, expected_params_to_tune",
    [
        (
            [
                Pipeline(
                    model=CatBoostPerSegmentModel(iterations=100),
                    transforms=[DateFlagsTransform(), LagTransform(in_column="target", lags=[1, 2, 3])],
                    horizon=3,
                ),
                Pipeline(model=ProphetModel(), transforms=[StandardScalerTransform()], horizon=3),
                Pipeline(model=NaiveModel(lag=3), horizon=3),
            ],
            {
                "pipelines.0.model.learning_rate": FloatDistribution(low=1e-4, high=0.5, log=True),
                "pipelines.0.model.depth": IntDistribution(low=1, high=11, step=1),
                "pipelines.0.model.l2_leaf_reg": FloatDistribution(low=0.1, high=200.0, log=True),
                "pipelines.0.model.random_strength": FloatDistribution(low=1e-05, high=10.0, log=True),
                "pipelines.0.transforms.0.day_number_in_week": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.day_number_in_month": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.day_number_in_year": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.week_number_in_month": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.week_number_in_year": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.month_number_in_year": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.season_number": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.year_number": CategoricalDistribution([False, True]),
                "pipelines.0.transforms.0.is_weekend": CategoricalDistribution([False, True]),
                "pipelines.1.model.seasonality_mode": CategoricalDistribution(["additive", "multiplicative"]),
                "pipelines.1.model.seasonality_prior_scale": FloatDistribution(low=1e-2, high=10, log=True),
                "pipelines.1.model.changepoint_prior_scale": FloatDistribution(low=1e-3, high=0.5, log=True),
                "pipelines.1.model.changepoint_range": FloatDistribution(low=0.8, high=0.95),
                "pipelines.1.model.holidays_prior_scale": FloatDistribution(low=1e-2, high=10, log=True),
                "pipelines.1.transforms.0.mode": CategoricalDistribution(["per-segment", "macro"]),
                "pipelines.1.transforms.0.with_mean": CategoricalDistribution([False, True]),
                "pipelines.1.transforms.0.with_std": CategoricalDistribution([False, True]),
            },
        )
    ],
)
def test_params_to_tune(pipelines, expected_params_to_tune):
    voting_ensemble_pipeline = VotingEnsemble(pipelines=pipelines)

    assert voting_ensemble_pipeline.params_to_tune() == expected_params_to_tune
