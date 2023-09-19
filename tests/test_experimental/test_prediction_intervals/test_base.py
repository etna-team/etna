import numpy as np
import pandas as pd
import pytest

from etna.ensembles import DirectEnsemble
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.models import CatBoostPerSegmentModel
from etna.models import NaiveModel
from etna.pipeline import AutoRegressivePipeline
from etna.pipeline import HierarchicalPipeline
from etna.pipeline import Pipeline
from etna.reconciliation import BottomUpReconciliator
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from tests.test_experimental.test_prediction_intervals.common import DummyPredictionIntervals


def run_base_pipeline_compat_check(ts, pipeline, expected_columns):
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    intervals_pipeline.fit(ts=ts)

    intervals_pipeline_pred = intervals_pipeline.forecast(prediction_interval=True)
    columns = intervals_pipeline_pred.df.columns.get_level_values("feature")

    assert len(expected_columns - set(columns)) == 0
    assert np.sum(intervals_pipeline_pred.df.isna().values) == 0


def get_naive_pipeline(horizon):
    return Pipeline(model=NaiveModel(), transforms=[], horizon=horizon)


def get_naive_pipeline_with_transforms(horizon):
    transforms = [AddConstTransform(in_column="target", value=1e6), DateFlagsTransform()]
    return Pipeline(model=NaiveModel(), transforms=transforms, horizon=horizon)


@pytest.fixture()
def naive_pipeline():
    return get_naive_pipeline(horizon=5)


@pytest.fixture()
def naive_pipeline_with_transforms():
    return get_naive_pipeline_with_transforms(horizon=5)


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


@pytest.mark.parametrize(
    "expected_columns",
    ({"target_lower", "target_upper"},),
)
@pytest.mark.parametrize("pipeline_name", ("naive_pipeline", "naive_pipeline_with_transforms"))
def test_forecast_intervals_exists(example_tsds, pipeline_name, expected_columns, request):
    pipeline = request.getfixturevalue(pipeline_name)

    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    intervals_pipeline.fit(ts=example_tsds)

    intervals_pipeline_pred = intervals_pipeline.forecast(prediction_interval=True)
    columns = intervals_pipeline_pred.df.columns.get_level_values("feature")

    assert len(expected_columns - set(columns)) == 0


@pytest.mark.parametrize(
    "expected_columns",
    ({"target", "target_lower", "target_upper"},),
)
@pytest.mark.parametrize(
    "pipeline",
    (
        Pipeline(model=NaiveModel(), horizon=1),
        AutoRegressivePipeline(model=NaiveModel(), horizon=1),
        HierarchicalPipeline(
            model=NaiveModel(),
            horizon=1,
            reconciliator=BottomUpReconciliator(target_level="market", source_level="product"),
        ),
    ),
)
def test_pipelines_forecast_intervals(product_level_constant_hierarchical_ts, pipeline, expected_columns):
    run_base_pipeline_compat_check(
        ts=product_level_constant_hierarchical_ts, pipeline=pipeline, expected_columns=expected_columns
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
    run_base_pipeline_compat_check(ts=example_tsds, pipeline=ensemble, expected_columns=expected_columns)


@pytest.mark.parametrize(
    "pipeline",
    (
        get_naive_pipeline(horizon=1),
        get_naive_pipeline_with_transforms(horizon=2),
        Pipeline(model=CatBoostPerSegmentModel()),
        AutoRegressivePipeline(model=CatBoostPerSegmentModel(), horizon=1),
        HierarchicalPipeline(
            model=NaiveModel(),
            horizon=1,
            reconciliator=BottomUpReconciliator(target_level="market", source_level="product"),
        ),
    ),
)
def test_default_params_to_tune(pipeline):
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)
    assert intervals_pipeline.params_to_tune() == pipeline.params_to_tune()


@pytest.mark.parametrize(
    "pipeline",
    (VotingEnsemble(pipelines=[get_naive_pipeline(horizon=1), get_naive_pipeline_with_transforms(horizon=1)]),),
)
def test_default_params_to_tune_error(pipeline):
    intervals_pipeline = DummyPredictionIntervals(pipeline=pipeline)

    with pytest.raises(NotImplementedError, match=f"{pipeline.__class__.__name__} doesn't support"):
        _ = intervals_pipeline.params_to_tune()
