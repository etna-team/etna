from torch.utils.data.sampler import RandomSampler


class SamplerWrapper:
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, data):
        if issubclass(self.sampler, RandomSampler):
            return self.sampler(data)
        else:
            weights = [sample["weight"] for sample in data]
            return self.sampler(weights)
