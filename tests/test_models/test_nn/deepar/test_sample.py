import pytest
from torch.utils.data import RandomSampler

from etna.models.nn.deepar_new.sampler import SamplerWrapper
from etna.models.nn.deepar_new.sampler import WeightedDeepARSampler


@pytest.mark.parametrize("sampler", [RandomSampler, WeightedDeepARSampler])
def test_sampler(sampler):
    sampler = SamplerWrapper(sampler)
    data = [{"weight": 1.0}, {"weight": 2.0}, {"weight": 3.0}]
    called_sampler = sampler(data)
    iterator = iter(called_sampler)
    next(iterator)
