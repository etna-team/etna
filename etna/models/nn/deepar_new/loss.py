from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.distributions import NegativeBinomial
    from torch.distributions import Normal
    from torch.nn.modules.loss import _Loss


class GaussianLoss(_Loss):
    """GaussianLoss is used to count loss in DeepAR model."""

    def __init__(self):
        """Init GaussianLoss."""
        super().__init__()

    @staticmethod
    def scale_params(loc, scale, weights):
        """Make transformation of predicted parameters of distribution.

        Parameters
        ----------
        loc:
            mean of Gaussian distribution.
        scale:
            standard deviation of Gaussian distribution.
        weights:
            weights used for transformation.

        Returns
        -------
        :
            transformed mean and standard deviation

        """
        mean = loc.clone()
        std = scale.clone()
        reshaped = [-1] + [1] * (loc.dim() - 1)
        weight = weights.reshape(reshaped).expand(loc.shape)
        mean *= weight
        std *= weight.abs()
        return mean, std

    def forward(self, inputs, loc, scale, weights):
        """Forward pass.

        Parameters
        ----------
        inputs:
            true target values
        loc:
            mean of Gaussian distribution.
        scale:
            standard deviation of Gaussian distribution.
        weights:
            weights used for transformation.

        Returns
        -------
        :
            loss

        """
        mean, std = self.scale_params(loc, scale, weights)
        distribution = Normal(loc=mean, scale=std)
        return -(distribution.log_prob(inputs)).sum()

    def sample(self, loc, scale, weights, n_samples):
        """Get samples from distribution.

        Parameters
        ----------
        loc:
            mean of Gaussian distribution.
        scale:
            standard deviation of Gaussian distribution.
        weights:
            weights used for transformation.
        n_samples:
            number of samples to generate from distribution

        Returns
        -------
        :
            samples from distribution

        """
        mean, std = self.scale_params(loc, scale, weights)
        distribution = Normal(loc=mean, scale=std)
        return distribution.loc if n_samples == 1 else distribution.sample()


class NegativeBinomialLoss(_Loss):
    """NegativeBinomialLoss is used to count loss in DeepAR model."""

    def __init__(self):
        """Init NegativeBinomialLoss."""
        super().__init__()

    @staticmethod
    def scale_params(loc, scale, weights):
        """Make transformation of predicted parameters of distribution.

        Parameters
        ----------
        loc:
            mean of NegativeBinomial distribution.
        scale:
            shape parameter of NegativeBinomial distribution.
        weights:
            weights used for transformation.

        Returns
        -------
        :
            number of successes until the experiment is stopped and success probability

        """
        mean = loc.clone()
        alpha = scale.clone()
        reshaped = [-1] + [1] * (loc.dim() - 1)
        weights = weights.reshape(reshaped).expand(loc.shape)
        total_count = torch.sqrt(weights) / alpha
        probs = 1 / (torch.sqrt(weights) * mean * alpha + 1)
        return total_count, probs

    def forward(self, inputs, loc, scale, weights):
        """Forward pass.

        Parameters
        ----------
        inputs:
            true target values
        loc:
            mean of NegativeBinomial distribution.
        scale:
            shape parameter of NegativeBinomial distribution.
        weights:
            weights used for transformation.

        Returns
        -------
        :
            lass

        """
        total_count, probs = self.scale_params(loc, scale, weights)
        distribution = NegativeBinomial(total_count=total_count, probs=probs)
        return -(distribution.log_prob(inputs)).sum()

    def sample(self, loc, scale, weights, n_samples):
        """Get samples from distribution.

        Parameters
        ----------
        loc:
            mean of NegativeBinomial distribution.
        scale:
            shape parameter of NegativeBinomial distribution.
        weights:
            weights used for transformation.
        n_samples:
            number of samples to generate from distribution

        Returns
        -------
        :
            samples from distribution

        """
        total_count, probs = self.scale_params(loc, scale, weights)
        distribution = NegativeBinomial(total_count=total_count, probs=probs)
        return distribution.mean if n_samples == 1 else distribution.sample()
