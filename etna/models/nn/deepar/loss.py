from abc import abstractmethod
from typing import Tuple

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.distributions import NegativeBinomial
    from torch.distributions import Normal
    from torch.nn.modules.loss import _Loss


class DeepARLoss(_Loss):
    """Base class to create any loss for DeepARModel."""

    @staticmethod
    @abstractmethod
    def scale_params(
        loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make transformation of predicted parameters of distribution.

        Parameters
        ----------
        loc:
            first parameter of distribution.
        scale:
            second parameter of distribution.
        weights:
            weights of the samples used for transformation.

        Returns
        -------
        :
            transformed parameters

        """
        pass

    @abstractmethod
    def forward(
        self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs:
            true target values
        loc:
            first parameter of distribution.
        scale:
            second parameter of distribution.
        weights:
            weights of the samples used for transformation.

        Returns
        -------
        :
            loss

        """
        pass

    @abstractmethod
    def sample(
        self, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, theoretical_mean: bool
    ) -> torch.Tensor:
        """Get samples from distribution.

        Parameters
        ----------
        loc:
            first parameter of distribution.
        scale:
            second parameter of distribution.
        weights:
            weights of the samples used for transformation.
        theoretical_mean:
            if True return theoretical_mean of distribution, else return sample from distribution

        Returns
        -------
        :
            samples from distribution

        """
        pass


class GaussianLoss(DeepARLoss):
    """Negative log likelihood loss for Gaussian distribution."""

    @staticmethod
    def scale_params(
        loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        mean = loc.clone()
        std = scale.clone()
        weights = weights.reshape(-1, 1, 1).expand(loc.shape)
        mean *= weights
        std *= weights.abs()
        return mean, std

    def forward(
        self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
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

    def sample(
        self, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, theoretical_mean: bool
    ) -> torch.Tensor:
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


class NegativeBinomialLoss(DeepARLoss):
    """Negative log likelihood loss for NegativeBinomial distribution."""

    @staticmethod
    def scale_params(
        loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        mean = loc.clone()
        alpha = scale.clone()
        weights = weights.reshape(-1, 1, 1).expand(loc.shape)
        total_count = torch.sqrt(weights) / alpha
        probs = 1 - (1 / (torch.sqrt(weights) * mean * alpha + 1))
        return total_count, probs

    def forward(
        self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
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

    def sample(
        self, loc: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, theoretical_mean: bool
    ) -> torch.Tensor:
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
