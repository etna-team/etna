from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.distributions import NegativeBinomial
    from torch.distributions import Normal
    from torch.nn.modules.loss import _Loss


class GaussianLoss(_Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def scale_params(loc, scale, weights):
        mean = loc
        std = scale
        if not weights.isnan().any():
            reshaped = [-1] + [1] * (loc.dim() - 1)
            weight = weights.reshape(reshaped).expand(loc.shape)
            mean *= weight
            std *= weight.abs()
        return mean, std

    def forward(self, inputs, loc, scale, weights):
        mean, std = self.scale_params(loc, scale, weights)
        distribution = Normal(loc=mean, scale=std)
        return -(distribution.log_prob(inputs)).sum()

    def sample(self, loc, scale, weights):
        mean, std = self.scale_params(loc, scale, weights)
        distribution = Normal(loc=mean, scale=std)
        return distribution.sample()


class NegativeBinomialLoss(_Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def scale_params(loc, scale, weights):
        alpha = loc
        mean = scale
        if not weights.isnan().any():
            reshaped = [-1] + [1] * (loc.dim() - 1)
            weights = weights.reshape(reshaped).expand(loc.shape)
            total_count = torch.sqrt(torch.tensor(weights)) / alpha
            probs = 1 / (torch.sqrt(torch.tensor(weights)) * mean * alpha + 1)
        else:
            total_count = 1 / alpha
            probs = 1 / (mean * alpha + 1)  # TODO scale them into [0, 1]
        return total_count, probs

    def forward(self, inputs, loc, scale, weights):
        total_count, probs = self.scale_params(loc, scale, weights)
        distribution = NegativeBinomial(total_count=total_count, probs=probs)
        return -(distribution.log_prob(inputs)).sum()

    def sample(self, loc, scale, weights):
        total_count, probs = self.scale_params(loc, scale, weights)
        distribution = NegativeBinomial(total_count=total_count, probs=probs)
        return distribution.sample()
