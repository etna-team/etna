from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.distributions import NegativeBinomial
    from torch.distributions import Normal
    from torch.nn.modules.loss import _Loss


class GaussianLoss(_Loss):
    """Negative log likelihood loss for Gaussian distribution."""

    @staticmethod
    def scale_params(loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor):
        """Make transformation of predicted parameters of distribution.

        Parameters
        ----------
        loc:
            mean of Gaussian distribution.
        scale:
            standard deviation of Gaussian distribution.
        weights:
            weights of the samples used for transformation.

        Returns
        -------
        :
            transformed mean and standard deviation

        """
        mean = loc.clone()  # (batch_size, encoder_length + decoder_length - 1, 1)
        std = scale.clone()  # (batch_size, encoder_length + decoder_length - 1, 1)
        reshaped = [-1] + [1] * (loc.dim() - 1)
        weights = weights.reshape(reshaped).expand(
            loc.shape
        )  # (batch_size) -> (batch_size, encoder_length + decoder_length - 1, 1)
        mean *= weights
        std *= weights.abs()
        return mean, std

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor):
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
            weights of the samples used for transformation.

        Returns
        -------
        :
            loss

        """
        mean, std = self.scale_params(loc=loc, scale=scale, weights=weights)
        distribution = Normal(loc=mean, scale=std)
        return -(distribution.log_prob(inputs)).mean()

    def sample(self, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, theoretical_mean: bool):
        """Get samples from distribution.

        Parameters
        ----------
        loc:
            mean of Gaussian distribution.
        scale:
            standard deviation of Gaussian distribution.
        weights:
            weights of the samples used for transformation.
        theoretical_mean:
            if True return theoretical_mean of distribution, else return sample from distribution

        Returns
        -------
        :
            samples from distribution

        """
        mean, std = self.scale_params(loc=loc, scale=scale, weights=weights)
        distribution = Normal(loc=mean, scale=std)
        return distribution.loc if theoretical_mean else distribution.sample()


class NegativeBinomialLoss(_Loss):
    """Negative log likelihood loss for NegativeBinomial distribution."""

    @staticmethod
    def scale_params(loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor):
        """Make transformation of predicted parameters of distribution.

        Parameters
        ----------
        loc:
            mean of NegativeBinomial distribution.
        scale:
            shape parameter of NegativeBinomial distribution.
        weights:
            weights of the samples used for transformation.

        Returns
        -------
        :
            number of successes until the experiment is stopped and success probability

        """
        mean = loc.clone()  # (batch_size, encoder_length + decoder_length - 1, 1)
        alpha = scale.clone()  # (batch_size, encoder_length + decoder_length - 1, 1)
        reshaped = [-1] + [1] * (loc.dim() - 1)
        weights = weights.reshape(reshaped).expand(
            loc.shape
        )  # (batch_size) -> (batch_size, encoder_length + decoder_length - 1, 1)
        total_count = torch.sqrt(weights) / alpha
        probs = 1 / (torch.sqrt(weights) * mean * alpha + 1)
        return total_count, probs

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor):
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
            weights of the samples used for transformation.

        Returns
        -------
        :
            lass

        """
        total_count, probs = self.scale_params(loc=loc, scale=scale, weights=weights)
        distribution = NegativeBinomial(total_count=total_count, probs=probs)
        return -(distribution.log_prob(inputs)).mean()

    def sample(self, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, theoretical_mean: bool):
        """Get samples from distribution.

        Parameters
        ----------
        loc:
            mean of NegativeBinomial distribution.
        scale:
            shape parameter of NegativeBinomial distribution.
        weights:
            weights of the samples used for transformation.
        theoretical_mean:
            if True return theoretical_mean of distribution, else return sample from distribution

        Returns
        -------
        :
            samples from distribution

        """
        total_count, probs = self.scale_params(loc=loc, scale=scale, weights=weights)
        distribution = NegativeBinomial(total_count=total_count, probs=probs)
        return distribution.mean if theoretical_mean else distribution.sample()
