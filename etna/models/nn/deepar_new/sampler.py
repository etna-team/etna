from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.utils.data.sampler import Sampler


class WeightedDeepARSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):  # TODO if passed both weighted sampler and scale=False
        weights = torch.tensor([sample["weight"] for sample in self.data])
        idx = torch.multinomial(weights, num_samples=len(self.data), replacement=True)
        return iter(idx)

    def __len__(self):
        return len(self.data)


class SamplerWrapper:
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, data):
        return self.sampler(data)
