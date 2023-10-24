import torch
from torch import nn
from torch.functional import F
from torch.nn import GaussianNLLLoss


class GaussianLoss(GaussianNLLLoss):
    def __init__(self):
        super().__init__(reduction="sum")

    def forward(self, inputs, mean, std):
        return super().forward(input=inputs, target=mean, var=std**2)

    @staticmethod
    def sample(mean, std):
        with torch.no_grad():
            return torch.normal(mean, std)


class NegativeBinomialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_logits(probs):
        eps = torch.finfo(probs.dtype).eps
        ps_clamped = probs.clamp(min=eps, max=1 - eps)
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)

    def forward(self, inputs, total_count, probs):
        logits = self._get_logits(probs)
        log_unnormalized_prob = total_count * F.logsigmoid(logits) + inputs * F.logsigmoid(logits)

        log_normalization = -torch.lgamma(total_count + inputs) + torch.lgamma(1.0 + inputs) + torch.lgamma(total_count)

        return -(log_unnormalized_prob - log_normalization).sum()

    def _gamma(self, total_count, probs):
        logits = self._get_logits(probs)
        return torch.distributions.Gamma(concentration=total_count, rate=torch.exp(-logits), validate_args=False)

    def sample(self, total_count, probs, sample_shape=torch.Size()):
        with torch.no_grad():
            rate = self._gamma(total_count, probs).sample(sample_shape=sample_shape)
            return torch.poisson(rate)
