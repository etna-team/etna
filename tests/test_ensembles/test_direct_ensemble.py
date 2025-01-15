from unittest.mock import MagicMock
from unittest.mock import Mock

import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_from_patterns_df
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.ensembles import DirectEnsemble
from etna.metrics import MAE
from etna.models import CatBoostPerSegmentModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import HierarchicalPipeline
from etna.pipeline import Pipeline
from etna.reconciliation import TopDownReconciliator
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import StandardScalerTransform
from tests.test_pipeline.utils import assert_pipeline_equals_loaded_original
from tests.test_pipeline.utils import assert_pipeline_forecast_raise_error_if_no_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts_with_prediction_intervals
from tests.test_pipeline.utils import assert_pipeline_forecasts_without_self_ts
from tests.test_pipeline.utils import assert_pipeline_predicts


@pytest.fixture
def direct_ensemble_pipeline() -> DirectEnsemble:
    ensemble = DirectEnsemble(
        pipelines=[
            Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=1),
            Pipeline(model=NaiveModel(lag=3), transforms=[], horizon=2),
        ]
    )
    return ensemble


@pytest.fixture
def naive_pipeline_top_down_market_7() -> Pipeline:
    """Generate pipeline with NaiveModel."""
    pipeline = HierarchicalPipeline(
        model=NaiveModel(),
        transforms=[],
        horizon=7,
        reconciliator=TopDownReconciliator(source_level="total", target_level="market", period=14, method="AHP"),
    )
    return pipeline


@pytest.fixture
def naive_pipeline_top_down_product_7() -> Pipeline:
    """Generate pipeline with NaiveModel."""
    pipeline = HierarchicalPipeline(
        model=NaiveModel(),
        transforms=[],
        horizon=7,
        reconciliator=TopDownReconciliator(source_level="total", target_level="product", period=14, method="AHP"),
    )
    return pipeline


@pytest.fixture
def direct_ensemble_hierarchical_pipeline(
    naive_pipeline_top_down_market_7, naive_pipeline_bottom_up_market_14
) -> DirectEnsemble:
    ensemble = DirectEnsemble(
        pipelines=[
            naive_pipeline_top_down_market_7,
            naive_pipeline_bottom_up_market_14,
        ]
    )
    return ensemble


@pytest.fixture
def direct_ensemble_mix_pipeline(naive_pipeline_top_down_product_7) -> DirectEnsemble:
    ensemble = DirectEnsemble(
        pipelines=[
            naive_pipeline_top_down_product_7,
            Pipeline(model=NaiveModel(), transforms=[], horizon=14),
        ]
    )
    return ensemble


@pytest.fixture
def simple_ts_train():
    df = generate_from_patterns_df(patterns=[[1, 3, 5], [2, 4, 6], [7, 9, 11]], periods=3, start_time="2000-01-01")
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def simple_ts_forecast():
    df = generate_from_patterns_df(patterns=[[5, 3], [6, 4], [11, 9]], periods=2, start_time="2000-01-04")
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


def test_get_horizon():
    ensemble = DirectEnsemble(pipelines=[Mock(horizon=1), Mock(horizon=2)])
    assert ensemble.horizon == 2


def test_get_horizon_raise_error_on_same_horizons():
    with pytest.raises(ValueError, match="All the pipelines should have pairwise different horizons."):
        _ = DirectEnsemble(pipelines=[Mock(horizon=1), Mock(horizon=1)])


@pytest.mark.parametrize("save_ts", [False, True])
def test_fit_saving_ts(direct_ensemble_pipeline, simple_ts_train, save_ts):
    direct_ensemble_pipeline.fit(simple_ts_train, save_ts=save_ts)

    if save_ts:
        assert direct_ensemble_pipeline.ts is simple_ts_train
    else:
        assert direct_ensemble_pipeline.ts is None


def test_forecast_values(direct_ensemble_pipeline, simple_ts_train, simple_ts_forecast):
    direct_ensemble_pipeline.fit(simple_ts_train)
    forecast = direct_ensemble_pipeline.forecast()
    pd.testing.assert_frame_equal(forecast.to_pandas(), simple_ts_forecast.to_pandas())


def test_predict_values(direct_ensemble_pipeline, simple_ts_train):
    smallest_pipeline = Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=1)
    direct_ensemble_pipeline.fit(simple_ts_train)
    smallest_pipeline.fit(simple_ts_train)
    prediction = direct_ensemble_pipeline.predict(
        ts=simple_ts_train, start_timestamp=simple_ts_train.index[1], end_timestamp=simple_ts_train.index[2]
    )
    expected_prediction = smallest_pipeline.predict(
        ts=simple_ts_train, start_timestamp=simple_ts_train.index[1], end_timestamp=simple_ts_train.index[2]
    )
    pd.testing.assert_frame_equal(prediction.to_pandas(), expected_prediction.to_pandas())


@pytest.mark.parametrize("load_ts", [True, False])
def test_save_load(load_ts, direct_ensemble_pipeline, example_tsds):
    assert_pipeline_equals_loaded_original(pipeline=direct_ensemble_pipeline, ts=example_tsds, load_ts=load_ts)


def test_forecast_raise_error_if_no_ts(direct_ensemble_pipeline, example_tsds):
    assert_pipeline_forecast_raise_error_if_no_ts(pipeline=direct_ensemble_pipeline, ts=example_tsds)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "direct_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "direct_ensemble_pipeline"),
    ],
)
def test_forecasts_without_self_ts(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_without_self_ts(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "direct_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "direct_ensemble_pipeline"),
    ],
)
def test_forecast_given_ts(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_given_ts(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "direct_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "direct_ensemble_pipeline"),
    ],
)
def test_forecast_given_ts_with_prediction_interval(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_forecasts_given_ts_with_prediction_intervals(pipeline=ensemble, ts=ts, horizon=ensemble.horizon)


@pytest.mark.parametrize(
    "ts_name, ensemble_name",
    [
        ("example_tsds", "direct_ensemble_pipeline"),
        ("example_tsds_int_timestamp", "direct_ensemble_pipeline"),
    ],
)
def test_predict(ts_name, ensemble_name, request):
    ts = request.getfixturevalue(ts_name)
    ensemble = request.getfixturevalue(ensemble_name)
    assert_pipeline_predicts(pipeline=ensemble, ts=ts, start_idx=20, end_idx=30)


def test_forecast_with_return_components_fails(example_tsds, direct_ensemble_pipeline):
    direct_ensemble_pipeline.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="Adding target components is not currently implemented!"):
        direct_ensemble_pipeline.forecast(return_components=True)


def test_predict_with_return_components_fails(example_tsds, direct_ensemble_pipeline):
    direct_ensemble_pipeline.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="Adding target components is not currently implemented!"):
        direct_ensemble_pipeline.predict(ts=example_tsds, return_components=True)


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
    pipeline_0.horizon = 3

    pipeline_1 = MagicMock()
    pipeline_1.params_to_tune.return_value = pipeline_1_tune_params
    pipeline_1.horizon = 7

    ensemble_pipeline = DirectEnsemble(pipelines=[pipeline_0, pipeline_1])

    assert ensemble_pipeline.params_to_tune() == expected_tune_params


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
                Pipeline(model=ProphetModel(), transforms=[StandardScalerTransform()], horizon=5),
                Pipeline(model=NaiveModel(lag=3), horizon=9),
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
    ensemble_pipeline = DirectEnsemble(pipelines=pipelines)

    assert ensemble_pipeline.params_to_tune() == expected_params_to_tune


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(direct_ensemble_pipeline, example_tsds, n_jobs: int):
    results = direct_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    for df in results:
        assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest_hierarchical_pipeline(
    direct_ensemble_hierarchical_pipeline, product_level_simple_hierarchical_ts_long_history: TSDataset, n_jobs: int
):
    results = direct_ensemble_hierarchical_pipeline.backtest(
        ts=product_level_simple_hierarchical_ts_long_history, metrics=[MAE()], n_jobs=n_jobs, n_folds=3
    )
    for df in results:
        assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest_mix_pipeline(
    direct_ensemble_mix_pipeline, product_level_simple_hierarchical_ts_long_history: TSDataset, n_jobs: int
):
    results = direct_ensemble_mix_pipeline.backtest(
        ts=product_level_simple_hierarchical_ts_long_history, metrics=[MAE()], n_jobs=n_jobs, n_folds=3
    )
    for df in results:
        assert isinstance(df, pd.DataFrame)
