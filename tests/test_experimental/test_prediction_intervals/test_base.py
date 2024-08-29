import pandas as pd
import pytest

from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.ensembles import DirectEnsemble
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.models import CatBoostPerSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import NaiveModel
from etna.models import SeasonalMovingAverageModel
from etna.pipeline import AutoRegressivePipeline
from etna.pipeline import HierarchicalPipeline
from etna.pipeline import Pipeline
from etna.pipeline.base import _DummyMetric
from etna.reconciliation import BottomUpReconciliator
from etna.transforms import DateFlagsTransform
from etna.transforms import DeseasonalityTransform
from tests.test_experimental.test_prediction_intervals.common import DummyPredictionIntervals
from tests.test_experimental.test_prediction_intervals.common import get_naive_pipeline
from tests.test_experimental.test_prediction_intervals.common import get_naive_pipeline_with_transforms
from tests.test_experimental.test_prediction_intervals.common import run_base_pipeline_compat_check
from tests.test_experimental.test_prediction_intervals.utils import assert_sampling_is_valid
from tests.test_pipeline.utils import assert_pipeline_equals_loaded_original


def test_pipeline_ref_initialized(naive_pipeline):
    intervals_pipeline = DummyPredictionIntervals(pipeline=naive_pipeline)

    assert hasattr(intervals_pipeline, "pipeline")
    assert intervals_pipeline.pipeline is naive_pipeline


def test_ts_property(naive_pipeline):
    intervals_pipeline = DummyPredictionIntervals(pipeline=naive_pipeline)

    assert hasattr(intervals_pipeline, "ts")
    assert intervals_pipeline.ts is naive_pipeline.ts


def test_predict_default_error(example_tsds, naive_pipeline):
    intervals_pipeline = DummyPredictionIntervals(pipeline=naive_pipeline)
    intervals_pipeline.fit(ts=example_tsds)

    with pytest.raises(NotImplementedError, match="In-sample sample prediction is not supported"):
        _ = intervals_pipeline.predict(ts=example_tsds)


@pytest.mark.parametrize("pipeline_name", ("naive_pipeline", "naive_pipeline_with_transforms"))
def test_pipeline_fit_forecast(example_tsds, pipeline_name, request):
    pipeline = request.getfixturevalue(pipeline_name)

    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)

    intervals_pipeline.fit(ts=example_tsds)

    intervals_pipeline_pred = intervals_pipeline.forecast(prediction_interval=False)
    pipeline_pred = pipeline.forecast(prediction_interval=False)

    pd.testing.assert_frame_equal(intervals_pipeline_pred.df, pipeline_pred.df)


@pytest.mark.parametrize("pipeline_name", ("naive_pipeline", "naive_pipeline_with_transforms"))
def test_forecast_with_fitted_pipeline(example_tsds, pipeline_name, request):
    pipeline = request.getfixturevalue(pipeline_name)

    pipeline.fit(ts=example_tsds)
    pipeline_pred = pipeline.forecast(prediction_interval=False)

    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    intervals_pipeline_pred = intervals_pipeline.forecast(prediction_interval=False)

    pd.testing.assert_frame_equal(intervals_pipeline_pred.df, pipeline_pred.df)


@pytest.mark.parametrize("pipeline_name", ("naive_pipeline", "naive_pipeline_with_transforms"))
def test_backtest(example_tsds, pipeline_name, request):
    pipeline = request.getfixturevalue(pipeline_name)
    _, pipeline_results, _ = pipeline.backtest(ts=example_tsds, metrics=[_DummyMetric()])

    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    _, intervals_pipeline_results, _ = intervals_pipeline.backtest(ts=example_tsds, metrics=[_DummyMetric()])

    pd.testing.assert_frame_equal(pipeline_results, intervals_pipeline_results)


@pytest.mark.parametrize("pipeline_name", ("naive_pipeline", "naive_pipeline_with_transforms"))
def test_get_historical_forecasts(example_tsds, pipeline_name, request):
    pipeline = request.getfixturevalue(pipeline_name)
    pipeline_results = pipeline.get_historical_forecasts(ts=example_tsds)

    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    intervals_pipeline_results = intervals_pipeline.get_historical_forecasts(ts=example_tsds)

    pd.testing.assert_frame_equal(pipeline_results, intervals_pipeline_results)


@pytest.mark.parametrize(
    "expected_columns",
    ({"target", "target_lower", "target_upper"},),
)
@pytest.mark.parametrize(
    "pipeline",
    (
        get_naive_pipeline(horizon=1),
        get_naive_pipeline_with_transforms(horizon=1),
        AutoRegressivePipeline(model=NaiveModel(), horizon=1),
        HierarchicalPipeline(
            model=NaiveModel(),
            horizon=1,
            reconciliator=BottomUpReconciliator(target_level="market", source_level="product"),
        ),
    ),
)
def test_pipelines_forecast_intervals(product_level_constant_hierarchical_ts, pipeline, expected_columns):
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    run_base_pipeline_compat_check(
        ts=product_level_constant_hierarchical_ts,
        intervals_pipeline=intervals_pipeline,
        expected_columns=expected_columns,
    )


@pytest.mark.parametrize(
    "expected_columns",
    ({"target", "target_lower", "target_upper"},),
)
@pytest.mark.parametrize(
    "ensemble",
    (
        DirectEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=2)]),
        VotingEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=1)]),
        StackingEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=1)]),
    ),
)
def test_ensembles_forecast_intervals(example_tsds, ensemble, expected_columns):
    intervals_pipeline = DummyPredictionIntervals(pipeline=ensemble)
    run_base_pipeline_compat_check(
        ts=example_tsds, intervals_pipeline=intervals_pipeline, expected_columns=expected_columns
    )


@pytest.mark.parametrize(
    "pipeline,expected_params_to_tune",
    (
        (
            Pipeline(
                model=SeasonalMovingAverageModel(), transforms=[DeseasonalityTransform(in_column="target", period=7)]
            ),
            {
                "pipeline.model.window": IntDistribution(low=1, high=10),
                "pipeline.transforms.0.model": CategoricalDistribution(["additive", "multiplicative"]),
                "width": FloatDistribution(low=-5.0, high=5.0),
            },
        ),
        (
            AutoRegressivePipeline(model=CatBoostPerSegmentModel(), transforms=[DateFlagsTransform()], horizon=1),
            {
                "pipeline.model.learning_rate": FloatDistribution(low=1e-4, high=0.5, log=True),
                "pipeline.model.depth": IntDistribution(low=1, high=11, step=1),
                "pipeline.model.l2_leaf_reg": FloatDistribution(low=0.1, high=200.0, log=True),
                "pipeline.model.random_strength": FloatDistribution(low=1e-05, high=10.0, log=True),
                "pipeline.transforms.0.day_number_in_week": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.day_number_in_month": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.day_number_in_year": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.week_number_in_month": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.week_number_in_year": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.month_number_in_year": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.season_number": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.year_number": CategoricalDistribution([False, True]),
                "pipeline.transforms.0.is_weekend": CategoricalDistribution([False, True]),
                "width": FloatDistribution(low=-5.0, high=5.0),
            },
        ),
        (
            HierarchicalPipeline(
                model=SeasonalMovingAverageModel(),
                transforms=[DeseasonalityTransform(in_column="target", period=7)],
                horizon=1,
                reconciliator=BottomUpReconciliator(target_level="market", source_level="product"),
            ),
            {
                "pipeline.model.window": IntDistribution(low=1, high=10),
                "pipeline.transforms.0.model": CategoricalDistribution(["additive", "multiplicative"]),
                "width": FloatDistribution(low=-5.0, high=5.0),
            },
        ),
    ),
)
def test_params_to_tune(pipeline, expected_params_to_tune):
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)

    params_to_tune = intervals_pipeline.params_to_tune()

    assert params_to_tune == expected_params_to_tune


@pytest.mark.parametrize(
    "pipeline",
    (
        Pipeline(model=LinearPerSegmentModel(), transforms=[DateFlagsTransform()]),
        AutoRegressivePipeline(model=LinearPerSegmentModel(), transforms=[DateFlagsTransform()], horizon=1),
        HierarchicalPipeline(
            model=LinearPerSegmentModel(),
            transforms=[DateFlagsTransform()],
            horizon=1,
            reconciliator=BottomUpReconciliator(target_level="market", source_level="product"),
        ),
    ),
)
def test_valid_params_sampling(product_level_constant_hierarchical_ts, pipeline):
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    assert_sampling_is_valid(intervals_pipeline=intervals_pipeline, ts=product_level_constant_hierarchical_ts)


@pytest.mark.parametrize("load_ts", (True, False))
@pytest.mark.parametrize(
    "pipeline",
    (
        Pipeline(model=LinearPerSegmentModel(), transforms=[DateFlagsTransform()]),
        AutoRegressivePipeline(model=LinearPerSegmentModel(), transforms=[DateFlagsTransform()], horizon=1),
        HierarchicalPipeline(
            model=LinearPerSegmentModel(),
            transforms=[DateFlagsTransform()],
            horizon=1,
            reconciliator=BottomUpReconciliator(target_level="total", source_level="market"),
        ),
        DirectEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=2)]),
        VotingEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=1)]),
        StackingEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=1)]),
    ),
)
def test_save_load(load_ts, pipeline, market_level_constant_hierarchical_ts_w_exog):
    ts = market_level_constant_hierarchical_ts_w_exog
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    assert_pipeline_equals_loaded_original(pipeline=intervals_pipeline, ts=ts, load_ts=load_ts)
