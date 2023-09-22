import pytest

from tests.test_experimental.test_prediction_intervals.common import get_naive_pipeline
from tests.test_experimental.test_prediction_intervals.common import get_naive_pipeline_with_transforms


@pytest.fixture()
def naive_pipeline():
    return get_naive_pipeline(horizon=5)


@pytest.fixture()
def naive_pipeline_with_transforms():
    return get_naive_pipeline_with_transforms(horizon=5)
