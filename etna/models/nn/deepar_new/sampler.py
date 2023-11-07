from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.utils.data.sampler import Sampler


class WeightedDeepARSampler(Sampler):
    """Samples batch elements with probabilities `1 + mean`, where mean is a mean of target values in batch."""

    def __init__(self, data, n_samples):
        self.data = data
        self.n_samples = n_samples

    def __iter__(self):
        weights = torch.tensor([sample["weight"] for sample in self.data])
        idx = torch.multinomial(1 / weights, num_samples=self.n_samples, replacement=False)  # TODO ok?
        return iter(idx)

    def __len__(self):
        return self.n_samples


class SamplerWrapper:
    """Wrapper for samplers."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, data):
        """Call given sampler.

        Parameters
        ----------
        data:
            data

        Returns
        -------
        :
            object of given sampler
        """
        return self.sampler(data)
