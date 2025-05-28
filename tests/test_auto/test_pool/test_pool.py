from copy import deepcopy

import pytest

from etna.auto.pool import Pool
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
