import os
from pathlib import Path

import pytest
from hydra_slayer import get_from_params
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
    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def expected_ts_increasing_integers():
    df = generate_ar_df(start_time="2001-01-11", periods=1, n_segments=2)
    df["target"] = [10.0] + [110.0]
    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.mark.smoke
def test_chronos_url(tmp_path):
    model_name = "chronos-t5-tiny.zip"
    model_dir = model_name.split(".zip")[0]
    url = f"http://etna-github-prod.cdn-tinkoff.ru/chronos/{model_name}"
    _ = ChronosModel(path_or_url=url, cache_dir=tmp_path)
    assert os.path.exists(tmp_path / f"{tmp_path}/{model_dir}")


@pytest.mark.smoke
def test_chronos_bolt_url(tmp_path):
    model_name = "chronos-bolt-tiny.zip"
    model_dir = model_name.split(".zip")[0]
    url = f"http://etna-github-prod.cdn-tinkoff.ru/chronos/{model_name}"
    _ = ChronosBoltModel(path_or_url=url, cache_dir=tmp_path)
    assert os.path.exists(tmp_path / f"{tmp_path}/{model_dir}")


@pytest.mark.smoke
def test_chronos_custom_cache_dir(tmp_path):
    path_or_url = "amazon/chronos-t5-tiny"
    model_name = path_or_url.split("/")[-1]
    _ = ChronosModel(path_or_url=path_or_url, cache_dir=tmp_path)
    assert os.path.exists(tmp_path / f"models--amazon--{model_name}")


@pytest.mark.smoke
def test_chronos_bolt_custom_cache_dir(tmp_path):
    path_or_url = "amazon/chronos-bolt-tiny"
    model_name = path_or_url.split("/")[-1]
    _ = ChronosBoltModel(path_or_url=path_or_url, cache_dir=tmp_path)
    assert os.path.exists(tmp_path / f"models--amazon--{model_name}")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=10),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10),
    ],
)
def test_context_size(model):
    assert model.context_size == 10


@pytest.mark.smoke
def test_chronos_get_model(example_tsds):
    model = ChronosModel(path_or_url="amazon/chronos-t5-tiny")
    assert isinstance(model.get_model(), ChronosModelForForecasting)


@pytest.mark.smoke
def test_chronos_bolt_get_model(example_tsds):
    model = ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny")
    assert isinstance(model.get_model(), ChronosBoltModelForForecasting)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [ChronosModel(path_or_url="amazon/chronos-t5-tiny"), ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny")],
)
def test_fit(example_tsds, model):
    model = ChronosModel(path_or_url="amazon/chronos-t5-tiny")
    model.fit(example_tsds)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [ChronosModel(path_or_url="amazon/chronos-t5-tiny"), ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny")],
)
def test_predict(example_tsds, model):
    with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented!"):
        model.predict(ts=example_tsds, prediction_size=1)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=20),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=20),
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
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=10, limit_prediction_length=False),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10, limit_prediction_length=False),
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
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=10, limit_prediction_length=True),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10, limit_prediction_length=True),
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
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=10, num_samples=5),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10),
    ],
)
def test_forecast(ts_increasing_integers, expected_ts_increasing_integers, model):
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    forecast = pipeline.forecast()
    assert_frame_equal(forecast._df, expected_ts_increasing_integers._df, atol=2)


@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=10, num_samples=3),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10),
    ],
)
@pytest.mark.parametrize("ts", ["example_tsds", "example_tsds_int_timestamp"])
def test_forecast_prediction_intervals(ts, model, request):
    quantiles = [0.1, 0.9]
    ts = request.getfixturevalue(ts)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles)
    forecast_df = forecast.to_pandas(flatten=True)
    assert isinstance(forecast, TSDataset)
    assert len(forecast.timestamps) == 1
    assert forecast.features == ["target", "target_0.1", "target_0.9"]
    assert forecast_df["target_0.9"].gt(forecast_df["target_0.1"]).all()


def test_chronos_bolt_forecast_prediction_intervals_unusual_quantiles(ts_increasing_integers):
    quantiles = [0.025, 0.975]
    model = ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10)
    pipeline = Pipeline(model=model, horizon=3)
    pipeline.fit(ts=ts_increasing_integers)
    with pytest.warns(
        UserWarning,
        match=f"Quantiles to be predicted .+ are not within the range of quantiles that Chronos-Bolt was trained on \(\[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\]\).",
    ):
        _ = pipeline.forecast(prediction_interval=True, quantiles=quantiles)


@pytest.mark.smoke
@pytest.mark.parametrize("ts", ["example_tsds", "example_tsds_int_timestamp"])
@pytest.mark.parametrize(
    "model",
    [
        ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=2),
        ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=2),
    ],
)
def test_forecast_without_fit(ts, model, request):
    ts = request.getfixturevalue(ts)
    pipeline = Pipeline(model=model, horizon=1)
    _ = pipeline.forecast(ts)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [ChronosModel(path_or_url="amazon/chronos-t5-tiny"), ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny")],
)
def test_forecast_fails_components(example_tsds, model):
    pipeline = Pipeline(model=model, horizon=1)
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        pipeline.forecast(ts=example_tsds, return_components=True)


@pytest.mark.smoke
def test_chronos_list_models():
    assert ChronosModel.list_models() == [
        "amazon/chronos-t5-tiny",
        "amazon/chronos-t5-mini",
        "amazon/chronos-t5-small",
        "amazon/chronos-t5-base",
        "amazon/chronos-t5-large",
    ]


@pytest.mark.smoke
def test_chronos_bolt_list_models():
    assert ChronosBoltModel.list_models() == [
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-bolt-mini",
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
    ]


@pytest.mark.smoke
def test_chronos_save_load(tmp_path, ts_increasing_integers):
    path = Path(tmp_path) / "tmp.zip"
    model = ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=10)
    model.save(path)
    loaded_model = ChronosModel.load(path)

    pipeline = Pipeline(model=loaded_model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    _ = pipeline.forecast()
    assert isinstance(loaded_model, ChronosModel)


@pytest.mark.smoke
def test_chronos_bolt_save_load(tmp_path, ts_increasing_integers):
    path = Path(tmp_path) / "tmp.zip"
    model = ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=10)
    model.save(path)
    loaded_model = ChronosBoltModel.load(path)

    pipeline = Pipeline(model=loaded_model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    _ = pipeline.forecast()
    assert isinstance(loaded_model, ChronosBoltModel)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model",
    [ChronosModel(path_or_url="amazon/chronos-t5-tiny"), ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny")],
)
def test_params_to_tune(model):
    assert len(model.params_to_tune()) == 0


@pytest.mark.parametrize(
    "config",
    [
        {
            "_target_": "etna.models.nn.ChronosModel",
            "path_or_url": "amazon/chronos-bolt-tiny",
        },
        {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "amazon/chronos-t5-tiny",
        },
    ],
)
def test_chainable_get_from_params(config):

    model = get_from_params(**config)
    config_new = model.to_dict()
    _ = get_from_params(**config_new)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_all_string_dtypes(dtype):
    _ = ChronosModel(path_or_url="amazon/chronos-t5-tiny", dtype=dtype)
    _ = ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", dtype=dtype)
