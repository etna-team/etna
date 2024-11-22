import os
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.libs.chronos.chronos import ChronosForecaster
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
def test_fail_unknown_model_name():
    with pytest.raises(NotImplementedError, match="Model chronos-t5-supertiny is not available."):
        _ = ChronosModel(model_name="chronos-t5-supertiny")


@pytest.mark.smoke
def test_custom_cache_dir():
    cache_dir = Path("chronos")
    _ = ChronosModel(model_name="chronos-t5-tiny", cache_dir=cache_dir)
    assert os.path.exists(cache_dir)


@pytest.mark.smoke
@pytest.mark.parametrize("encoder_length", [1, 2])
def test_context_size(example_tsds, encoder_length):
    model = ChronosModel(model_name="chronos-t5-tiny")
    assert model.context_size == encoder_length


@pytest.mark.smoke
def test_get_model(example_tsds):
    model = ChronosModel(model_name="chronos-t5-tiny")
    assert isinstance(model.get_model(), ChronosForecaster)


@pytest.mark.smoke
def test_fit(example_tsds):
    model = ChronosModel(model_name="chronos-t5-tiny")
    model.fit(example_tsds)


@pytest.mark.smoke
def test_predict(example_tsds):
    model = ChronosModel(model_name="chronos-t5-tiny")
    with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented!"):
        model.predict(ts=example_tsds, prediction_size=1)


@pytest.mark.smoke
def test_forecast_warns_big_context_size(ts_increasing_integers):
    model = ChronosModel(model_name="chronos-t5-tiny", encoder_length=20)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    with pytest.warns(UserWarning, match="Actual length of a dataset is less that context size."):
        _ = pipeline.forecast()


def test_forecast(ts_increasing_integers, expected_ts_increasing_integers):
    model = ChronosModel(model_name="chronos-t5-tiny", encoder_length=10, num_samples=5, batch_size=2)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    forecast = pipeline.forecast()
    assert_frame_equal(forecast.df, expected_ts_increasing_integers.df, atol=2)


@pytest.mark.parametrize("ts", ["example_tsds", "example_tsds_int_timestamp"])
def test_forecast_prediction_intervals(ts, request):
    ts = request.getfixturevalue(ts)
    model = ChronosModel(model_name="chronos-t5-tiny", encoder_length=10, num_samples=1, batch_size=2)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.forecast(prediction_interval=True)
    assert isinstance(forecast, TSDataset)
    assert len(forecast.index) == 1
    assert len(forecast.features) == 3


@pytest.mark.smoke
@pytest.mark.parametrize("ts", ["example_tsds", "example_tsds_int_timestamp"])
def test_forecast_without_fit(ts, request):
    ts = request.getfixturevalue(ts)
    model = ChronosModel(model_name="chronos-t5-tiny", encoder_length=2)
    pipeline = Pipeline(model=model, horizon=1)
    _ = pipeline.forecast(ts)


@pytest.mark.smoke
def test_forecast_fails_components(example_tsds):
    model = ChronosModel(model_name="chronos-t5-tiny")
    pipeline = Pipeline(model=model, horizon=1)
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        pipeline.forecast(ts=example_tsds, return_components=True)


@pytest.mark.smoke
def test_list_models():
    assert ChronosModel.list_models() == [
        "chronos-t5-tiny",
        "chronos-t5-mini",
        "chronos-t5-small",
        "chronos-t5-base",
        "chronos-t5-large",
    ]


@pytest.mark.smoke
def test_save():
    model = ChronosModel(model_name="chronos-t5-tiny")
    assert model.save(".") is None


@pytest.mark.smoke
def test_load():
    model = ChronosModel(model_name="chronos-t5-tiny")
    assert model.load(".") is None


@pytest.mark.smoke
def test_params_to_tune(example_tsds):
    model = ChronosModel(model_name="chronos-t5-tiny")
    assert len(model.params_to_tune()) == 0
