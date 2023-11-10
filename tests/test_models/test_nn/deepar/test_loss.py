import pytest
import torch
from pytorch_lightning import seed_everything
from torch.distributions import NegativeBinomial
from torch.distributions import Normal

from etna.models.nn.deepar_new.loss import GaussianLoss
from etna.models.nn.deepar_new.loss import NegativeBinomialLoss


@pytest.mark.parametrize(
    "loss,distribution,loc,scale,weights",
    [
        (GaussianLoss(), Normal, torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])),
        (GaussianLoss(), Normal, torch.tensor([10.0]), torch.tensor([100.0]), torch.tensor([1.0])),
        (NegativeBinomialLoss(), NegativeBinomial, torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0])),
        (NegativeBinomialLoss(), NegativeBinomial, torch.tensor([0.6]), torch.tensor([0.1]), torch.tensor([1.0])),
    ],
)
def test_sample_mean(loss, distribution, loc, scale, weights):
    seed_everything(0)
    mean = loss.sample(loc=loc, scale=scale, weights=weights, theoretical_mean=True)
    scaled_params = loss.scale_params(loc, scale, weights)
    seed_everything(0)
    expected_mean = distribution(scaled_params[0], scaled_params[1]).mean
    torch.testing.assert_close(mean, expected_mean, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "loss,distribution,loc,scale,weights",
    [
        (GaussianLoss(), Normal, torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])),
        (GaussianLoss(), Normal, torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0])),
        (NegativeBinomialLoss(), NegativeBinomial, torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0])),
        (NegativeBinomialLoss(), NegativeBinomial, torch.tensor([10.0]), torch.tensor([0.2]), torch.tensor([1.0])),
    ],
)
def test_sample_random(loss, distribution, loc, scale, weights, n_samples=200):
    seed_everything(0)
    samples = torch.tensor([])
    for i in range(n_samples):
        sample = loss.sample(loc=loc, scale=scale, weights=weights, theoretical_mean=False)
        samples = torch.concat((samples, sample), dim=0)
    scaled_params = loss.scale_params(loc, scale, weights)
    seed_everything(0)
    distribution = distribution(scaled_params[0], scaled_params[1])
    expected_samples = torch.tensor([])
    for i in range(n_samples):
        expected_sample = distribution.sample()
        expected_samples = torch.concat((expected_samples, expected_sample), dim=0)
    torch.testing.assert_close(samples, expected_samples, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(torch.mean(samples, dim=0, keepdim=True), distribution.mean, atol=0.1, rtol=1e-10)


@pytest.mark.parametrize(
    "loss,loc,scale,weights,target,expected_loss",
    [
        (
            GaussianLoss(),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor(1.4189),
        ),
        (
            GaussianLoss(),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor(2.9189),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([2.0]),
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor(2.4142),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([2.0]),
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor(4.3113),
        ),
    ],
)
def test_forward(loss, loc, scale, weights, target, expected_loss):

    real_loss = loss(target, loc, scale, weights)
    print(real_loss)
    torch.testing.assert_close(real_loss, expected_loss, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "loss,inputs,expected_scaled_params",
    [
        (
            GaussianLoss(),
            (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0])),
            (torch.tensor([1.0]), torch.tensor([2.0])),
        ),
        (
            GaussianLoss(),
            (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([2.0])),
            (torch.tensor([2.0]), torch.tensor([4.0])),
        ),
        (
            NegativeBinomialLoss(),
            (torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0])),
            (torch.tensor([0.5]), torch.tensor([0.2])),
        ),
        (
            NegativeBinomialLoss(),
            (torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([4.0])),
            (torch.tensor([1.0]), torch.tensor([1 / 9])),
        ),
    ],
)
def test_scale_params(loss, inputs, expected_scaled_params):
    loc, scale, weights = inputs
    scaled_params = loss.scale_params(loc, scale, weights)
    torch.testing.assert_close(scaled_params, expected_scaled_params, atol=1e-4, rtol=1e-4)
