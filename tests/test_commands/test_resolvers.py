import pytest
from hydra_slayer import get_from_params

from etna.auto.pool.utils import fill_template


def construct_pipeline(config, horizon=7):
    config = fill_template(config, {"horizon": horizon})
    pipeline = get_from_params(**config)
    return pipeline


@pytest.fixture
def config_with_shift():
    return {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.LinearPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.LagTransform", "in_column": "target", "lags": "${shift:${horizon},[1, 2]}"}
        ],
    }


@pytest.fixture
def config_with_mult():
    return {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.LinearPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.LagTransform", "in_column": "target", "lags": ["${mult:${horizon},2}"]}
        ],
    }


@pytest.fixture
def config_with_concat():
    return {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.LinearPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.LagTransform", "in_column": "${concat:tar,get}", "lags": ["${horizon}"]}
        ],
    }


def test_shift(config_with_shift):
    pipeline = construct_pipeline(config_with_shift)
    assert pipeline.transforms[0].lags == [8, 9]


def test_mult(config_with_mult):
    pipeline = construct_pipeline(config_with_mult)
    assert pipeline.transforms[0].lags == [14]


def test_concat(config_with_concat):
    pipeline = construct_pipeline(config_with_concat)
    assert pipeline.transforms[0].in_column == "target"
