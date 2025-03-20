from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from etna.ensembles import DirectEnsemble
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.models import NaiveModel
from etna.pipeline import AutoRegressivePipeline
from etna.pipeline import HierarchicalPipeline
from etna.pipeline import Pipeline
from etna.prediction_intervals import EmpiricalPredictionIntervals
from etna.reconciliation import BottomUpReconciliator
from tests.test_prediction_intervals.common import get_arima_pipeline
from tests.test_prediction_intervals.common import get_catboost_pipeline
from tests.test_prediction_intervals.common import get_naive_pipeline
from tests.test_prediction_intervals.common import get_naive_pipeline_with_transforms
from tests.test_prediction_intervals.common import run_base_pipeline_compat_check


@pytest.mark.parametrize("stride", (-1, 0))
def test_invalid_stride_parameter_error(stride):
    with pytest.raises(ValueError, match="Parameter `stride` must be positive!"):
        EmpiricalPredictionIntervals(pipeline=Pipeline(model=NaiveModel()), stride=stride)


@pytest.mark.parametrize("coverage", (-3, -1))
def test_invalid_coverage_parameter_error(coverage):
    with pytest.raises(ValueError, match="Parameter `coverage` must be non-negative"):
        EmpiricalPredictionIntervals(pipeline=Pipeline(model=NaiveModel()), coverage=coverage)


@pytest.mark.parametrize("pipeline_name", ("naive_pipeline", "naive_pipeline_with_transforms"))
def test_pipeline_fit_forecast_without_intervals(example_tsds, pipeline_name, request):
    pipeline = request.getfixturevalue(pipeline_name)

    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=pipeline)

    intervals_pipeline.fit(ts=example_tsds)

    intervals_pipeline_pred = intervals_pipeline.forecast(prediction_interval=False)
    pipeline_pred = pipeline.forecast(prediction_interval=False)

    pd.testing.assert_frame_equal(intervals_pipeline_pred._df, pipeline_pred._df)


@pytest.mark.parametrize("stride", (2, 5, 10))
@pytest.mark.parametrize("expected_columns", ({"target", "target_lower", "target_upper"},))
def test_valid_strides(example_tsds, expected_columns, stride):
    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=Pipeline(model=NaiveModel(), horizon=5), stride=stride)
    run_base_pipeline_compat_check(
        ts=example_tsds, intervals_pipeline=intervals_pipeline, expected_columns=expected_columns
    )


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
def test_pipelines_forecast_intervals_exist(product_level_constant_hierarchical_ts, pipeline, expected_columns):
    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=pipeline)
    run_base_pipeline_compat_check(
        ts=product_level_constant_hierarchical_ts,
        intervals_pipeline=intervals_pipeline,
        expected_columns=expected_columns,
    )


@pytest.mark.parametrize("pipeline", (get_arima_pipeline(horizon=5),))
def test_forecast_prediction_intervals_is_used(example_tsds, pipeline):
    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=pipeline)
    intervals_pipeline._forecast_prediction_interval = MagicMock()

    intervals_pipeline.fit(ts=example_tsds)
    intervals_pipeline.forecast(prediction_interval=True)
    intervals_pipeline._forecast_prediction_interval.assert_called()


@pytest.mark.parametrize(
    "pipeline",
    (
        get_naive_pipeline(horizon=5),
        get_naive_pipeline_with_transforms(horizon=5),
        AutoRegressivePipeline(model=NaiveModel(), horizon=5),
        get_catboost_pipeline(horizon=5),
        get_arima_pipeline(horizon=5),
    ),
)
def test_pipelines_forecast_intervals_valid(example_tsds, pipeline):
    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=pipeline, include_forecast=True)
    intervals_pipeline.fit(ts=example_tsds)

    prediction = intervals_pipeline.forecast(prediction_interval=True)
    assert np.all(prediction[:, :, "target_lower"].values <= prediction[:, :, "target"].values)
    assert np.all(prediction[:, :, "target"].values <= prediction[:, :, "target_upper"].values)


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
def test_ensembles_forecast_intervals_exist(example_tsds, ensemble, expected_columns):
    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=ensemble)
    run_base_pipeline_compat_check(
        ts=example_tsds, intervals_pipeline=intervals_pipeline, expected_columns=expected_columns
    )


@pytest.mark.parametrize(
    "ensemble",
    (
        DirectEnsemble(pipelines=[get_naive_pipeline(horizon=5), get_naive_pipeline_with_transforms(horizon=6)]),
        VotingEnsemble(pipelines=[get_naive_pipeline(horizon=5), get_naive_pipeline_with_transforms(horizon=5)]),
        StackingEnsemble(pipelines=[get_naive_pipeline(horizon=5), get_naive_pipeline_with_transforms(horizon=5)]),
    ),
)
def test_ensembles_forecast_intervals_valid(example_tsds, ensemble):
    intervals_pipeline = EmpiricalPredictionIntervals(pipeline=ensemble, include_forecast=True)
    intervals_pipeline.fit(ts=example_tsds)

    prediction = intervals_pipeline.forecast(prediction_interval=True)
    assert np.all(prediction[:, :, "target_lower"].values <= prediction[:, :, "target"].values)
    assert np.all(prediction[:, :, "target"].values <= prediction[:, :, "target_upper"].values)
