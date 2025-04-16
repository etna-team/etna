from copy import deepcopy
from typing import Optional
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.metrics import MAE
from etna.metrics import MetricAggregationMode
from etna.models import CatBoostMultiSegmentModel
from etna.models import CatBoostPerSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline import AutoRegressivePipeline
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import FourierTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from tests.test_pipeline.utils import assert_pipeline_equals_loaded_original
from tests.test_pipeline.utils import assert_pipeline_forecast_raise_error_if_no_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts_with_prediction_intervals
from tests.test_pipeline.utils import assert_pipeline_forecasts_without_self_ts
from tests.test_pipeline.utils import assert_pipeline_predicts

DEFAULT_METRICS = [MAE(mode=MetricAggregationMode.per_segment)]


def test_forecast_additional_columns_warning(example_tsds):
    transform = LagTransform(lags=[3, 4], in_column="target")
    transformed_ts = transform.fit_transform(ts=example_tsds)

    pipeline = AutoRegressivePipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(lags=[5, 6], in_column="target")], horizon=2
    )
    pipeline.fit(transformed_ts)

    # different behavior to Pipeline, but results in the same error.
    with pytest.raises(KeyError, match=".* not in index"):
        _ = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])


@pytest.mark.parametrize(
    "ts_name,feature",
    (
        ("example_tsds", "target"),  # different behavior to Pipeline, modifications in target are preserved.
        ("outliers_tsds_without_missing", "exog"),  # different behavior to Pipeline, modifications in exog are ignored.
    ),
)
def test_forecast_prior_modifications(ts_name, feature, request, example_tsds):
    ts = request.getfixturevalue(ts_name)
    transform = AddConstTransform(value=1000, in_column=feature)
    transformed_ts = transform.fit_transform(ts=ts)

    pipeline = AutoRegressivePipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(lags=[5, 6], in_column="target")], horizon=2
    )
    pipeline.fit(transformed_ts)

    _ = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])


@pytest.mark.parametrize("save_ts", [False, True])
def test_fit(example_tsds, save_ts):
    """Test that AutoRegressivePipeline pipeline makes fit without failing."""
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform()]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=5, step=1)
    pipeline.fit(example_tsds, save_ts=save_ts)


@pytest.mark.parametrize("save_ts", [False, True])
def test_fit_saving_ts(example_tsds, save_ts):
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform()]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=5, step=1)
    pipeline.fit(example_tsds, save_ts=save_ts)

    if save_ts:
        assert pipeline.ts is example_tsds
    else:
        assert pipeline.ts is None


def fake_forecast(ts: TSDataset, prediction_size: Optional[int] = None, return_components: bool = False):
    df = ts.to_pandas()

    df.loc[:, pd.IndexSlice[:, "target"]] = 0
    if prediction_size is not None:
        df = df.iloc[-prediction_size:]

    ts._df = df

    return TSDataset(df=df, freq=ts.freq)


def spy_decorator(method_to_decorate):
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


@pytest.mark.parametrize(
    "model_class", [NonPredictionIntervalContextIgnorantAbstractModel, PredictionIntervalContextIgnorantAbstractModel]
)
def test_private_forecast_context_ignorant_model(model_class, example_tsds):
    # we should do it this way because we want not to change behavior but have ability to inspect calls
    # source: https://stackoverflow.com/a/41599695
    make_future = spy_decorator(TSDataset.make_future)
    model = MagicMock(spec=model_class)
    model.forecast.side_effect = fake_forecast

    with patch.object(TSDataset, "make_future", make_future):
        pipeline = AutoRegressivePipeline(model=model, horizon=5, step=1)
        pipeline.fit(example_tsds)
        _ = pipeline._forecast(ts=example_tsds, return_components=False)

    assert make_future.mock.call_count == 5
    make_future.mock.assert_called_with(future_steps=pipeline.step, transforms=())
    assert model.forecast.call_count == 5
    model.forecast.assert_called_with(ts=ANY, return_components=False)


@pytest.mark.parametrize(
    "model_class", [NonPredictionIntervalContextRequiredAbstractModel, PredictionIntervalContextRequiredAbstractModel]
)
def test_private_forecast_context_required_model(model_class, example_tsds):
    # we should do it this way because we want not to change behavior but have ability to inspect calls
    # source: https://stackoverflow.com/a/41599695
    make_future = spy_decorator(TSDataset.make_future)
    model = MagicMock(spec=model_class)
    model.context_size = 1
    model.forecast.side_effect = fake_forecast

    with patch.object(TSDataset, "make_future", make_future):
        pipeline = AutoRegressivePipeline(model=model, horizon=5, step=1)
        pipeline.fit(example_tsds)
        _ = pipeline._forecast(ts=example_tsds, return_components=False)

    assert make_future.mock.call_count == 5
    make_future.mock.assert_called_with(future_steps=pipeline.step, transforms=(), tail_steps=model.context_size)
    assert model.forecast.call_count == 5
    model.forecast.assert_called_with(ts=ANY, prediction_size=pipeline.step, return_components=False)


def test_forecast_columns(example_reg_tsds):
    """Test that AutoRegressivePipeline generates all the columns."""
    original_ts = deepcopy(example_reg_tsds)
    horizon = 5

    # make predictions in AutoRegressivePipeline
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform(is_weekend=True)]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    pipeline.fit(example_reg_tsds)
    forecast_pipeline = pipeline.forecast()

    # generate all columns
    original_ts.fit_transform(transforms)

    assert set(forecast_pipeline._df.columns) == set(original_ts._df.columns)

    # make sure that all values are filled
    assert forecast_pipeline.to_pandas().isna().sum().sum() == 0

    # check regressor values
    assert forecast_pipeline[:, :, "regressor_exog_weekend"].equals(
        original_ts._df_exog.loc[forecast_pipeline.timestamps, pd.IndexSlice[:, "regressor_exog_weekend"]]
    )


def test_forecast_one_step(example_tsds):
    """Test that AutoRegressivePipeline gets predictions one by one if step is equal to 1."""
    original_ts = deepcopy(example_tsds)
    horizon = 5

    # make predictions in AutoRegressivePipeline
    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10), LagTransform(in_column="target", lags=[1])]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    # make predictions manually
    df = original_ts.to_pandas()
    original_ts.fit_transform(transforms)
    model = LinearPerSegmentModel()
    model.fit(original_ts)
    for i in range(horizon):
        cur_ts = TSDataset(df, freq=original_ts.freq)
        # these transform don't fit and we can fit_transform them at each step
        cur_ts.transform(transforms)
        cur_forecast_ts = cur_ts.make_future(1, transforms=transforms)
        cur_future_ts = model.forecast(cur_forecast_ts)
        cur_future_ts.inverse_transform(transforms)
        to_add_df = cur_future_ts.to_pandas()
        df = pd.concat([df, to_add_df[df.columns]])

    forecast_manual = TSDataset(df.tail(horizon), freq=original_ts.freq)
    assert np.all(forecast_pipeline[:, :, "target"] == forecast_manual[:, :, "target"])


@pytest.mark.parametrize("horizon, step", ((1, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (20, 1), (20, 2), (20, 3)))
def test_forecast_multi_step(example_tsds, horizon, step):
    """Test that AutoRegressivePipeline gets correct number of predictions if step is more than 1."""
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[step])]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=step)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    assert forecast_pipeline.size()[0] == horizon


def test_forecast_prediction_interval_interface(example_tsds):
    """Test the forecast interface with prediction intervals."""
    pipeline = AutoRegressivePipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=[1])], horizon=5, step=1
    )
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_forecast_with_fit_transforms(example_tsds):
    """Test that AutoRegressivePipeline can work with transforms that need fitting."""
    horizon = 5

    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), LinearTrendTransform(in_column="target")]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    pipeline.fit(example_tsds)
    pipeline.forecast()


def test_backtest_with_n_jobs(big_example_tsdf: TSDataset):
    """Check that AutoRegressivePipeline.backtest gives the same results in case of single and multiple jobs modes."""
    # create a pipeline
    pipeline = AutoRegressivePipeline(
        model=CatBoostPerSegmentModel(),
        transforms=[LagTransform(in_column="target", lags=[1, 2, 3, 4, 5], out_column="regressor_lag_feature")],
        horizon=7,
        step=1,
    )

    # run forecasting
    ts1 = deepcopy(big_example_tsdf)
    ts2 = deepcopy(big_example_tsdf)
    pipeline_1 = deepcopy(pipeline)
    pipeline_2 = deepcopy(pipeline)

    forecast_ts_list_1 = pipeline_1.backtest(ts=ts1, n_jobs=1, metrics=DEFAULT_METRICS)["forecasts"]

    forecast_ts_list_2 = pipeline_2.backtest(ts=ts2, n_jobs=3, metrics=DEFAULT_METRICS)["forecasts"]

    # compare the results taking into account NaNs
    for forecast_ts_1, forecast_ts_2 in zip(forecast_ts_list_1, forecast_ts_list_2):
        pd.testing.assert_frame_equal(forecast_ts_1.to_pandas(), forecast_ts_2.to_pandas())


def test_backtest_forecasts_sanity(step_ts: TSDataset):
    """Check that AutoRegressivePipeline.backtest gives correct forecasts according to the simple case."""
    ts, expected_metrics_df, expected_forecast_df = step_ts
    pipeline = AutoRegressivePipeline(model=NaiveModel(), horizon=5, step=1)
    backtest_result = pipeline.backtest(ts, metrics=[MAE()], n_folds=3)

    metrics_df = backtest_result["metrics"]
    forecast_ts_list = backtest_result["forecasts"]
    forecast_df = pd.concat(
        [
            TSDataset.to_dataset(forecast_ts.to_pandas(flatten=True).assign(fold_number=num_fold))
            for num_fold, forecast_ts in enumerate(forecast_ts_list)
        ]
    )

    assert np.all(metrics_df.reset_index(drop=True) == expected_metrics_df)
    assert np.all(forecast_df == expected_forecast_df)


def test_get_historical_forecasts_sanity(step_ts: TSDataset):
    """Check that AutoRegressivePipeline.get_historical_forecasts gives correct forecasts according to the simple case."""
    ts, expected_metrics_df, expected_forecast_df = step_ts
    pipeline = AutoRegressivePipeline(model=NaiveModel(), horizon=5, step=1)
    forecast_ts_list = pipeline.get_historical_forecasts(ts, n_folds=3)
    forecast_df = pd.concat(
        [
            TSDataset.to_dataset(forecast_ts.to_pandas(flatten=True).assign(fold_number=num_fold))
            for num_fold, forecast_ts in enumerate(forecast_ts_list)
        ]
    )

    assert np.all(forecast_df == expected_forecast_df)


def test_predict_values(example_tsds):
    original_ts = deepcopy(example_tsds)

    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    predictions_pipeline = pipeline.predict(ts=original_ts)

    original_ts.fit_transform(transforms)
    model.fit(original_ts)
    predictions_manual = model.predict(original_ts)
    predictions_manual.inverse_transform(transforms)

    pd.testing.assert_frame_equal(predictions_pipeline.to_pandas(), predictions_manual.to_pandas())


@pytest.mark.parametrize("load_ts", [True, False])
@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (SeasonalMovingAverageModel(window=2, seasonality=7), []),
        (SARIMAXModel(), []),
        (ProphetModel(), []),
    ],
)
def test_save_load(load_ts, model, transforms, example_tsds):
    horizon = 3
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    assert_pipeline_equals_loaded_original(pipeline=pipeline, ts=example_tsds, load_ts=load_ts)


def test_forecast_raise_error_if_no_ts(example_tsds):
    pipeline = AutoRegressivePipeline(model=NaiveModel(), horizon=5)
    assert_pipeline_forecast_raise_error_if_no_ts(pipeline=pipeline, ts=example_tsds)


@pytest.mark.parametrize(
    "ts_name, model, transforms",
    [
        (
            "example_tsds",
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds",
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds", SARIMAXModel(), []),
        ("example_tsds", ProphetModel(), []),
        (
            "example_tsds_int_timestamp",
            CatBoostMultiSegmentModel(iterations=100),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds_int_timestamp",
            LinearPerSegmentModel(),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds_int_timestamp", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds_int_timestamp", SARIMAXModel(), []),
    ],
)
def test_forecasts_without_self_ts(ts_name, model, transforms, request):
    ts = request.getfixturevalue(ts_name)
    horizon = 3
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_without_self_ts(pipeline=pipeline, ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, model, transforms",
    [
        (
            "example_tsds",
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds",
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds", SARIMAXModel(), []),
        ("example_tsds", ProphetModel(), []),
        (
            "example_tsds_int_timestamp",
            CatBoostMultiSegmentModel(iterations=100),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds_int_timestamp",
            LinearPerSegmentModel(),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds_int_timestamp", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds_int_timestamp", SARIMAXModel(), []),
    ],
)
def test_forecast_given_ts(ts_name, model, transforms, request):
    ts = request.getfixturevalue(ts_name)
    horizon = 3
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_given_ts(pipeline=pipeline, ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, model, transforms",
    [
        (
            "example_tsds",
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds",
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds", SARIMAXModel(), []),
        ("example_tsds", ProphetModel(), []),
        (
            "example_tsds_int_timestamp",
            CatBoostMultiSegmentModel(iterations=100),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds_int_timestamp",
            LinearPerSegmentModel(),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds_int_timestamp", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds_int_timestamp", SARIMAXModel(), []),
    ],
)
def test_forecast_given_ts_with_prediction_interval(ts_name, model, transforms, request):
    ts = request.getfixturevalue(ts_name)
    horizon = 3
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_given_ts_with_prediction_intervals(pipeline=pipeline, ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, model, transforms",
    [
        (
            "example_tsds",
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds",
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds", SARIMAXModel(), []),
        ("example_tsds", ProphetModel(), []),
        (
            "example_tsds_int_timestamp",
            CatBoostMultiSegmentModel(iterations=100),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            "example_tsds_int_timestamp",
            LinearPerSegmentModel(),
            [FourierTransform(period=7, order=1), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        ("example_tsds_int_timestamp", SeasonalMovingAverageModel(window=2, seasonality=7), []),
        ("example_tsds_int_timestamp", SARIMAXModel(), []),
    ],
)
def test_predict(ts_name, model, transforms, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=7)
    assert_pipeline_predicts(pipeline=pipeline, ts=ts, start_idx=50, end_idx=70)


@pytest.mark.parametrize(
    "model_fixture",
    (
        "non_prediction_interval_context_ignorant_dummy_model",
        "non_prediction_interval_context_required_dummy_model",
        "prediction_interval_context_ignorant_dummy_model",
        "prediction_interval_context_required_dummy_model",
    ),
)
def test_forecast_return_components(
    example_tsds, model_fixture, request, expected_component_a=10, expected_component_b=90
):
    model = request.getfixturevalue(model_fixture)
    pipeline = AutoRegressivePipeline(model=model, horizon=10)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(return_components=True)
    assert sorted(forecast.target_components_names) == sorted(["target_component_a", "target_component_b"])

    target_components_df = TSDataset.to_flatten(forecast.get_target_components())
    assert (target_components_df["target_component_a"] == expected_component_a).all()
    assert (target_components_df["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize(
    "model_fixture",
    (
        "non_prediction_interval_context_ignorant_dummy_model",
        "non_prediction_interval_context_required_dummy_model",
        "prediction_interval_context_ignorant_dummy_model",
        "prediction_interval_context_required_dummy_model",
    ),
)
def test_predict_return_components(
    example_tsds, model_fixture, request, expected_component_a=20, expected_component_b=180
):
    model = request.getfixturevalue(model_fixture)
    pipeline = AutoRegressivePipeline(model=model, horizon=10)
    pipeline.fit(example_tsds)
    forecast = pipeline.predict(ts=example_tsds, return_components=True)
    assert sorted(forecast.target_components_names) == sorted(["target_component_a", "target_component_b"])

    target_components_df = TSDataset.to_flatten(forecast.get_target_components())
    assert (target_components_df["target_component_a"] == expected_component_a).all()
    assert (target_components_df["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize(
    "model, transforms, expected_params_to_tune",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
            {
                "model.learning_rate": FloatDistribution(low=1e-4, high=0.5, log=True),
                "model.depth": IntDistribution(low=1, high=11, step=1),
                "model.l2_leaf_reg": FloatDistribution(low=0.1, high=200.0, log=True),
                "model.random_strength": FloatDistribution(low=1e-05, high=10.0, log=True),
                "transforms.0.day_number_in_week": CategoricalDistribution([False, True]),
                "transforms.0.day_number_in_month": CategoricalDistribution([False, True]),
                "transforms.0.day_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.week_number_in_month": CategoricalDistribution([False, True]),
                "transforms.0.week_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.month_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.season_number": CategoricalDistribution([False, True]),
                "transforms.0.year_number": CategoricalDistribution([False, True]),
                "transforms.0.is_weekend": CategoricalDistribution([False, True]),
            },
        ),
    ],
)
def test_params_to_tune(model, transforms, expected_params_to_tune):
    horizon = 3
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon)

    obtained_params_to_tune = pipeline.params_to_tune()

    assert obtained_params_to_tune == expected_params_to_tune
