from copy import deepcopy

import pytest

from etna.auto.pool import Pool
from etna.auto.pool import PoolGenerator
from etna.auto.pool.templates import NO_FREQ_SUPER_FAST
from etna.datasets import TSDataset
from etna.pipeline import Pipeline


def test_generate_config():
    pipelines = Pool.no_freq_super_fast.value.generate(horizon=1, generate_params={})
    assert len(pipelines) == len(NO_FREQ_SUPER_FAST)


@pytest.mark.filterwarnings("ignore: Actual length of a dataset is less that context size.")
@pytest.mark.parametrize(
    "ts,generate_params",
    [("example_reg_tsds", {}), ("ts_with_external_timestamp", {"timestamp_column": "external_timestamp"})],
)
def test_default_pool_fit_predict(ts, generate_params, request):
    ts = request.getfixturevalue(ts)
    horizon = 7
    pipelines = Pool.no_freq_super_fast.value.generate(horizon=horizon, generate_params={})

    def fit_predict(pipeline: Pipeline, ts: TSDataset) -> TSDataset:
        pipeline.fit(deepcopy(ts))
        ts_forecast = pipeline.forecast()
        return ts_forecast

    ts_forecasts = [fit_predict(pipeline, ts) for pipeline in pipelines]

    for ts_forecast in ts_forecasts:
        assert len(ts_forecast.to_pandas()) == horizon


@pytest.mark.parametrize(
    "generate_params",
    [{}, {"timestamp_column": "external_timestamp", "chronos_device": "cpu"}],
)
def test_generate_params(
    generate_params,
    pool=(
        {
            "_target_": "etna.pipeline.Pipeline",
            "horizon": "${__aux__.horizon}",
            "model": {
                "_target_": "etna.models.nn.ChronosBoltModel",
                "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-tiny.zip",
                "encoder_length": 2048,
                "device": "${__aux__.chronos_device}",
                "batch_size": 128,
            },
            "transforms": [],
        },
        {
            "_target_": "etna.pipeline.Pipeline",
            "horizon": "${__aux__.horizon}",
            "model": {
                "_target_": "etna.models.ProphetModel",
                "seasonality_mode": "additive",
                "timestamp_column": "${__aux__.timestamp_column}",
            },
            "transforms": [],
        },
    ),
):
    horizon = 7
    default_params = {"timestamp_column": None, "chronos_device": "auto", "timesfm_device": "gpu"}

    generator = PoolGenerator(pool)
    pipelines = generator.generate(horizon=horizon, generate_params=generate_params)

    expected_params = default_params | generate_params

    assert pipelines[0].model.device == expected_params["chronos_device"]
    assert pipelines[1].model.timestamp_column == expected_params["timestamp_column"]


def test_not_required_generate_params(
    pool=(
        {
            "_target_": "etna.pipeline.Pipeline",
            "horizon": "${__aux__.horizon}",
            "model": {
                "_target_": "etna.models.ProphetModel",
                "seasonality_mode": "${__aux__.mode}",
                "timestamp_column": "${__aux__.timestamp_column}",
            },
            "transforms": [],
        },
    ),
):
    horizon = 7
    generate_params = {"timestamp_column": None}

    generator = PoolGenerator(pool)
    with pytest.raises(ValueError, match="Interpolation key .+ not found"):
        _ = generator.generate(horizon=horizon, generate_params=generate_params)


def test_horizon_collision(
    pool=(
        {
            "_target_": "etna.pipeline.Pipeline",
            "horizon": "${__aux__.horizon}",
            "model": {
                "_target_": "etna.models.ProphetModel",
                "seasonality_mode": "additive",
            },
            "transforms": [],
        },
    ),
):
    expected_horizon = 7
    generate_params = {"horizon": 1}

    generator = PoolGenerator(pool)
    pipe = generator.generate(horizon=expected_horizon, generate_params=generate_params)
    assert expected_horizon == pipe[0].horizon
