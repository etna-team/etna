import os
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.libs.chronos.chronos import ChronosModelForForecasting
from etna.libs.chronos.chronos_bolt import ChronosBoltModelForForecasting
from etna.models.nn import ChronosBoltModel
from etna.models.nn import ChronosModel
from etna.pipeline import Pipeline


@pytest.fixture
def ts_increasing_integers():
    df = generate_ar_df(start_time="2001-01-01", periods=10, n_segments=2)
    df["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_ts_increasing_integers():
    df = generate_ar_df(start_time="2001-01-11", periods=1, n_segments=2)
    df["target"] = [10.0] + [110.0]
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.smoke
def test_chronos_fail_unknown_model_name():
    with pytest.raises(NotImplementedError, match="Model chronos-t5-supertiny is not available."):
        _ = ChronosModel(model_name="chronos-t5-supertiny")


@pytest.mark.smoke
def test_chronos_bolt_fail_unknown_model_name():
    with pytest.raises(NotImplementedError, match="Model chronos-bolt-supertiny is not available."):
        _ = ChronosBoltModel(model_name="chronos-bolt-supertiny")


@pytest.mark.smoke
def test_chronos_custom_cache_dir():
    cache_dir = Path("chronos")
    _ = ChronosModel(model_name="chronos-t5-tiny", cache_dir=cache_dir)
    assert os.path.exists(cache_dir)


@pytest.mark.smoke
def test_chronos_bolt_custom_cache_dir():
    cache_dir = Path("chronos")
    _ = ChronosBoltModel(model_name="chronos-bolt-tiny", cache_dir=cache_dir)
    assert os.path.exists(cache_dir)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=10),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10),
    ],
)
def test_context_size(model):
    assert model.context_size == 10


@pytest.mark.smoke
def test_chronos_get_model(example_tsds):
    model = ChronosModel(model_name="chronos-t5-tiny")
    assert isinstance(model.get_model(), ChronosModelForForecasting)


@pytest.mark.smoke
def test_chronos_bolt_get_model(example_tsds):
    model = ChronosBoltModel(model_name="chronos-bolt-tiny")
    assert isinstance(model.get_model(), ChronosBoltModelForForecasting)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model", [ChronosModel(model_name="chronos-t5-tiny"), ChronosBoltModel(model_name="chronos-bolt-tiny")]
)
def test_fit(example_tsds, model):
    model = ChronosModel(model_name="chronos-t5-tiny")
    model.fit(example_tsds)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model", [ChronosModel(model_name="chronos-t5-tiny"), ChronosBoltModel(model_name="chronos-bolt-tiny")]
)
def test_predict(example_tsds, model):
    with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented!"):
        model.predict(ts=example_tsds, prediction_size=1)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=20),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=20),
    ],
)
def test_forecast_warns_big_context_size(ts_increasing_integers, model):
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    with pytest.warns(UserWarning, match="Actual length of a dataset is less that context size."):
        _ = pipeline.forecast()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=10, limit_prediction_length=False),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10, limit_prediction_length=False),
    ],
)
def test_forecast_warns_big_prediction_length(ts_increasing_integers, model):
    if isinstance(model.get_model(), ChronosModelForForecasting):
        config_prediction_length = model.get_model().config.prediction_length
    else:
        config_prediction_length = model.get_model().chronos_config.prediction_length
    pipeline = Pipeline(model=model, horizon=65)
    pipeline.fit(ts_increasing_integers)
    with pytest.warns(UserWarning, match=f"We recommend keeping prediction length <= {config_prediction_length}."):
        _ = pipeline.forecast()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=10, limit_prediction_length=True),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10, limit_prediction_length=True),
    ],
)
def test_forecast_error_big_prediction_length(ts_increasing_integers, model):
    if isinstance(model.get_model(), ChronosModelForForecasting):
        config_prediction_length = model.get_model().config.prediction_length
    else:
        config_prediction_length = model.get_model().chronos_config.prediction_length
    pipeline = Pipeline(model=model, horizon=65)
    pipeline.fit(ts_increasing_integers)
    with pytest.raises(ValueError, match=f"We recommend keeping prediction length <= {config_prediction_length}. "):
        _ = pipeline.forecast()


@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=10, num_samples=5),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10),
    ],
)
def test_forecast(ts_increasing_integers, expected_ts_increasing_integers, model):
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    forecast = pipeline.forecast()
    assert_frame_equal(forecast.df, expected_ts_increasing_integers.df, atol=2)


@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=10, num_samples=1),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10),
    ],
)
@pytest.mark.parametrize("ts", ["example_tsds", "example_tsds_int_timestamp"])
def test_forecast_prediction_intervals(ts, model, request):
    quantiles = [0.1, 0.9]
    ts = request.getfixturevalue(ts)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles)
    assert isinstance(forecast, TSDataset)
    assert len(forecast.index) == 1
    assert len(forecast.features) == 3


def test_chronos_bolt_forecast_prediction_intervals_unusual_quantiles(ts_increasing_integers):
    quantiles = [0.025, 0.975]
    model = ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10)
    pipeline = Pipeline(model=model, horizon=3)
    pipeline.fit(ts=ts_increasing_integers)
    with pytest.warns(
        UserWarning,
        match=f"Quantiles to be predicted ({quantiles}) are not within the range of quantiles that Chronos-Bolt was trained on ({model.get_model().chronos_config.quantiles}).",
    ):
        _ = pipeline.forecast(prediction_interval=True, quantiles=quantiles)


@pytest.mark.smoke
@pytest.mark.parametrize("ts", ["example_tsds", "example_tsds_int_timestamp"])
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(model_name="chronos-t5-tiny", encoder_length=2),
        ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=2),
    ],
)
def test_forecast_without_fit(ts, model, request):
    ts = request.getfixturevalue(ts)
    pipeline = Pipeline(model=model, horizon=1)
    _ = pipeline.forecast(ts)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model", [ChronosModel(model_name="chronos-t5-tiny"), ChronosBoltModel(model_name="chronos-bolt-tiny")]
)
def test_forecast_fails_components(example_tsds, model):
    pipeline = Pipeline(model=model, horizon=1)
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        pipeline.forecast(ts=example_tsds, return_components=True)


@pytest.mark.smoke
def test_chronos_list_models():
    assert ChronosModel.list_models() == [
        "chronos-t5-tiny",
        "chronos-t5-mini",
        "chronos-t5-small",
        "chronos-t5-base",
        "chronos-t5-large",
    ]


@pytest.mark.smoke
def test_chronos_bolt_list_models():
    assert ChronosBoltModel.list_models() == [
        "chronos-bolt-tiny",
        "chronos-bolt-mini",
        "chronos-bolt-small",
        "chronos-bolt-base",
    ]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model", [ChronosModel(model_name="chronos-t5-tiny"), ChronosBoltModel(model_name="chronos-bolt-tiny")]
)
def test_save(model):
    assert model.save(".") is None


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model", [ChronosModel(model_name="chronos-t5-tiny"), ChronosBoltModel(model_name="chronos-bolt-tiny")]
)
def test_load(model):
    assert model.load(".") is None


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model", [ChronosModel(model_name="chronos-t5-tiny"), ChronosBoltModel(model_name="chronos-bolt-tiny")]
)
def test_params_to_tune(model):
    assert len(model.params_to_tune()) == 0
