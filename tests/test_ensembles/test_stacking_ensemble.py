from copy import deepcopy
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.ensembles.stacking_ensemble import StackingEnsemble
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


@pytest.mark.parametrize("input_cv,true_cv", ([(2, 2)]))
def test_cv_pass(naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline, input_cv, true_cv):
    """Check that StackingEnsemble._validate_cv works correctly in case of valid cv parameter."""
    ensemble = StackingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], n_folds=input_cv)
    assert ensemble.n_folds == true_cv


@pytest.mark.parametrize("input_cv", ([0]))
def test_cv_fail_wrong_number(naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline, input_cv):
    """Check that StackingEnsemble._validate_cv works correctly in case of wrong number for cv parameter."""
    with pytest.raises(ValueError, match="Folds number should be a positive number, 0 given"):
        _ = StackingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], n_folds=input_cv)


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, None),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week"],
            {"regressor_lag_feature_10", "regressor_dateflag_day_number_in_week"},
        ),
    ),
)
def test_features_to_use(
    forecasts_df: List[pd.DataFrame],
    naive_featured_pipeline_1,
    naive_featured_pipeline_2,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._get_features_to_use works correctly."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    obtained_features = ensemble._filter_features_to_use(forecasts_df)
    assert obtained_features == expected_features


@pytest.mark.parametrize("features_to_use", (["regressor_lag_feature_10"]))
def test_features_to_use_wrong_format(
    forecasts_df: List[pd.DataFrame],
    naive_featured_pipeline_1,
    naive_featured_pipeline_2,
    features_to_use: Union[None, Literal[all], List[str]],
):
    """Check that StackingEnsemble._get_features_to_use raises warning in case of wrong format."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    with pytest.warns(UserWarning, match="Feature list is passed in the wrong format."):
        _ = ensemble._filter_features_to_use(forecasts_df)


@pytest.mark.parametrize("features_to_use", ([["unknown_feature"]]))
def test_features_to_use_not_found(
    forecasts_df: List[pd.DataFrame],
    naive_featured_pipeline_1,
    naive_featured_pipeline_2,
    features_to_use: Union[None, Literal[all], List[str]],
):
    """Check that StackingEnsemble._get_features_to_use raises warning in case of unavailable features."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    with pytest.warns(UserWarning, match=f"Features {set(features_to_use)} are not found and will be dropped!"):
        _ = ensemble._filter_features_to_use(forecasts_df)


@pytest.mark.filterwarnings("ignore: Features {'unknown'} are not found and will be dropped!")
@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week", "unknown"],
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
    ),
)
def test_make_features(
    example_tsds,
    forecasts_df: List[pd.DataFrame],
    targets,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._make_features returns X,y with all the expected columns
    and which are compatible with the sklearn interface.
    """
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    ).fit(example_tsds)
    x, y = ensemble._make_features(ts=example_tsds, forecasts=forecasts_df, train=True)
    features = set(x.columns.get_level_values("feature"))
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert features == expected_features
    assert (y == targets).all()


@pytest.mark.parametrize("save_ts", [False, True])
def test_fit_saving_ts(example_tsds, naive_pipeline_1, naive_pipeline_2, save_ts):
    ensemble = StackingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(example_tsds, save_ts=save_ts)

    if save_ts:
        assert ensemble.ts is example_tsds
    else:
        assert ensemble.ts is None


@pytest.mark.filterwarnings("ignore: Features {'unknown'} are not found and will be dropped!")
@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week", "unknown"],
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
    ),
)
def test_forecast_interface(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble.forecast returns TSDataset of correct length, containing all the expected columns"""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    ).fit(example_tsds)
    forecast = ensemble.forecast()
    features = set(forecast.features) - {"target"}
    assert isinstance(forecast, TSDataset)
    assert forecast.size()[0] == HORIZON
    assert features == expected_features


@pytest.mark.filterwarnings("ignore: Features {'unknown'} are not found and will be dropped!")
@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week", "unknown"],
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
    ),
)
def test_predict_interface(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble.predict returns TSDataset of correct length, containing all the expected columns"""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    ).fit(example_tsds)
    start_idx = 20
    end_idx = 30
    prediction = ensemble.predict(
        ts=example_tsds,
        start_timestamp=example_tsds.timestamps[start_idx],
        end_timestamp=example_tsds.timestamps[end_idx],
    )
    features = set(prediction.features) - {"target"}
    assert isinstance(prediction, TSDataset)
    assert prediction.size()[0] == end_idx - start_idx + 1
    assert features == expected_features


def test_forecast_prediction_interval_interface(example_tsds, naive_ensemble: StackingEnsemble):
    """Test the forecast interface with prediction intervals."""
    naive_ensemble.fit(example_tsds)
    forecast = naive_ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_forecast_calls_process_forecasts(example_tsds: TSDataset, naive_ensemble):
    naive_ensemble.fit(ts=example_tsds)
    naive_ensemble._process_forecasts = MagicMock()

    result = naive_ensemble._forecast(ts=example_tsds, return_components=False)

    naive_ensemble._process_forecasts.assert_called_once()
    assert result == naive_ensemble._process_forecasts.return_value


def test_predict_calls_process_forecasts(example_tsds: TSDataset, naive_ensemble):
    naive_ensemble.fit(ts=example_tsds)
    naive_ensemble._process_forecasts = MagicMock()

    result = naive_ensemble._predict(
        ts=example_tsds,
        start_timestamp=example_tsds.timestamps[20],
        end_timestamp=example_tsds.timestamps[30],
        prediction_interval=False,
        quantiles=(),
        return_components=False,
    )

    naive_ensemble._process_forecasts.assert_called_once()
    assert result == naive_ensemble._process_forecasts.return_value


def test_forecast_sanity(weekly_period_ts: Tuple["TSDataset", "TSDataset"], naive_ensemble: StackingEnsemble):
    """Check that StackingEnsemble.forecast forecast correct values"""
    train, test = weekly_period_ts
    ensemble = naive_ensemble.fit(train)
    forecast = ensemble.forecast()
    mae = MAE("macro")
    np.allclose(mae(test, forecast), 0)


def test_multiprocessing_ensembles(
    simple_tsdf,
    catboost_pipeline: Pipeline,
    prophet_pipeline: Pipeline,
    naive_pipeline_1: Pipeline,
    naive_pipeline_2: Pipeline,
):
    """Check that StackingEnsemble works the same in case of multi and single jobs modes."""
    pipelines = [catboost_pipeline, prophet_pipeline, naive_pipeline_1, naive_pipeline_2]
    single_jobs_ensemble = StackingEnsemble(pipelines=deepcopy(pipelines), n_jobs=1)
    multi_jobs_ensemble = StackingEnsemble(pipelines=deepcopy(pipelines), n_jobs=3)

    single_jobs_ensemble.fit(ts=deepcopy(simple_tsdf))
    multi_jobs_ensemble.fit(ts=deepcopy(simple_tsdf))

    single_jobs_forecast = single_jobs_ensemble.forecast()
    multi_jobs_forecast = multi_jobs_ensemble.forecast()

    assert (single_jobs_forecast._df == multi_jobs_forecast._df).all().all()


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(stacking_ensemble_pipeline: StackingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that backtest works with StackingEnsemble."""
    results = stacking_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    check_backtest_return_type(results, StackingEnsemble)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest_hierarchical_pipeline(
    stacking_ensemble_hierarchical_pipeline: StackingEnsemble,
    product_level_simple_hierarchical_ts_long_history: TSDataset,
    n_jobs: int,
):
    """Check that backtest works with StackingEnsemble of hierarchical pipelines."""
    results = stacking_ensemble_hierarchical_pipeline.backtest(
        ts=product_level_simple_hierarchical_ts_long_history, metrics=[MAE()], n_jobs=n_jobs, n_folds=3
    )
    check_backtest_return_type(results, StackingEnsemble)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest_mix_pipeline(
    stacking_ensemble_mix_pipeline: StackingEnsemble,
    product_level_simple_hierarchical_ts_long_history: TSDataset,
    n_jobs: int,
):
    """Check that backtest works with StackingEnsemble of pipeline and hierarchical pipeline."""
    results = stacking_ensemble_mix_pipeline.backtest(
        ts=product_level_simple_hierarchical_ts_long_history, metrics=[MAE()], n_jobs=n_jobs, n_folds=3
    )
    check_backtest_return_type(results, StackingEnsemble)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_get_historical_forecasts(stacking_ensemble_pipeline: StackingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that get_historical_forecasts works with StackingEnsemble."""
    n_folds = 3
    forecast_ts_list = stacking_ensemble_pipeline.get_historical_forecasts(
        ts=example_tsds, n_jobs=n_jobs, n_folds=n_folds
    )
    assert isinstance(forecast_ts_list, List)
    for forecast_ts in forecast_ts_list:
        assert forecast_ts.size()[0] == stacking_ensemble_pipeline.horizon


@pytest.mark.parametrize("load_ts", [True, False])
def test_save_load(stacking_ensemble_pipeline, example_tsds, load_ts):
    assert_pipeline_equals_loaded_original(pipeline=stacking_ensemble_pipeline, ts=example_tsds, load_ts=load_ts)


def test_forecast_raise_error_if_no_ts(stacking_ensemble_pipeline, example_tsds):
    assert_pipeline_forecast_raise_error_if_no_ts(pipeline=stacking_ensemble_pipeline, ts=example_tsds)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "stacking_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "stacking_ensemble_pipeline_int_timestamp"),
    ],
)
def test_forecasts_without_self_ts(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_without_self_ts(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "stacking_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "stacking_ensemble_pipeline_int_timestamp"),
    ],
)
def test_forecast_given_ts(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_given_ts(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "stacking_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "stacking_ensemble_pipeline_int_timestamp"),
    ],
)
def test_forecast_given_ts_with_prediction_interval(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_given_ts_with_prediction_intervals(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "stacking_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "stacking_ensemble_pipeline_int_timestamp"),
    ],
)
def test_predict(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_predicts(pipeline=ensemble, ts=ts, start_idx=20, end_idx=30)


def test_forecast_with_return_components_fails(example_tsds, naive_ensemble):
    naive_ensemble.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="Adding target components is not currently implemented!"):
        naive_ensemble.forecast(return_components=True)


def test_predict_with_return_components_fails(example_tsds, naive_ensemble):
    naive_ensemble.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="Adding target components is not currently implemented!"):
        naive_ensemble.predict(ts=example_tsds, return_components=True)


@pytest.mark.parametrize("n_jobs", (1, 4))
def test_ts_with_segment_named_target(
    ts_with_segment_named_target: TSDataset, stacking_ensemble_pipeline: StackingEnsemble, n_jobs: int
):
    results = stacking_ensemble_pipeline.backtest(
        ts=ts_with_segment_named_target, metrics=[MAE()], n_jobs=n_jobs, n_folds=5
    )
    check_backtest_return_type(results, StackingEnsemble)


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
    pipeline_0.horizon = 5

    pipeline_1 = MagicMock()
    pipeline_1.params_to_tune.return_value = pipeline_1_tune_params
    pipeline_1.horizon = 5

    ensemble_pipeline = StackingEnsemble(pipelines=[pipeline_0, pipeline_1])

    assert ensemble_pipeline.params_to_tune() == expected_tune_params


@pytest.mark.parametrize(
    "pipelines, expected_params_to_tune",
    [
        (
            [
                Pipeline(
                    model=CatBoostPerSegmentModel(iterations=100),
                    transforms=[DateFlagsTransform(), LagTransform(in_column="target", lags=[1, 2, 3])],
                    horizon=5,
                ),
                Pipeline(model=ProphetModel(), transforms=[StandardScalerTransform()], horizon=5),
                Pipeline(model=NaiveModel(lag=3), horizon=5),
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
    ensemble_pipeline = StackingEnsemble(pipelines=pipelines)

    assert ensemble_pipeline.params_to_tune() == expected_params_to_tune
