import pytest
import torch
from pytorch_lightning import seed_everything

from etna.models.nn.deepar.loss import GaussianLoss
from etna.models.nn.deepar.loss import NegativeBinomialLoss


@pytest.mark.parametrize(
    "loss,loc,scale,weights,expected_mean",
    [
        (
            GaussianLoss(),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[0.0]]]),
        ),
        (
            GaussianLoss(),
            torch.tensor([[[10.0]]]),
            torch.tensor([[[100.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[10.0]]]),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[0.1]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[2.0]]]),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([[[0.6]]]),
            torch.tensor([[[0.1]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[0.6]]]),
        ),
    ],
)
def test_sample_mean(loss, loc, scale, weights, expected_mean):
    mean = loss.sample(loc=loc, scale=scale, weights=weights, theoretical_mean=True)
    torch.testing.assert_close(mean, expected_mean)


@pytest.mark.parametrize(
    "loss,loc,scale,weights",
    [
        (GaussianLoss(), torch.tensor([[[0.0]]]), torch.tensor([[[1.0]]]), torch.tensor([[[1.0]]])),
        (GaussianLoss(), torch.tensor([[[1.0]]]), torch.tensor([[[2.0]]]), torch.tensor([[[1.0]]])),
        (NegativeBinomialLoss(), torch.tensor([[[2.0]]]), torch.tensor([[[0.2]]]), torch.tensor([[[1.0]]])),
        (NegativeBinomialLoss(), torch.tensor([[[10.0]]]), torch.tensor([[[0.2]]]), torch.tensor([[[1.0]]])),
    ],
)
def test_sample_random(loss, loc, scale, weights, n_samples=200):
    seed_everything(0)
    samples = torch.concat(
        [loss.sample(loc=loc, scale=scale, weights=weights, theoretical_mean=False) for _ in range(n_samples)], dim=0
    )
    expected_mean = loss.sample(loc=loc, scale=scale, weights=weights, theoretical_mean=True)
    torch.testing.assert_close(expected_mean, torch.mean(samples, dim=0, keepdim=True), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "loss,loc,scale,weights,target,expected_loss",
    [
        (
            GaussianLoss(),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor(1.4189),
        ),
        (
            GaussianLoss(),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor(2.9189),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor(1.7210),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor(2.2318),
        ),
    ],
)
def test_forward(loss, loc, scale, weights, target, expected_loss):

    real_loss = loss(target, loc, scale, weights)
    torch.testing.assert_close(real_loss, expected_loss, atol=1e-10, rtol=1e-3)


@pytest.mark.parametrize(
    "loss, loc, scale, weights, expected_scaled_loc, expected_scaled_scale",
    [
        (
            GaussianLoss(),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[2.0]]]),
        ),
        (
            GaussianLoss(),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[4.0]]]),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[0.5]]]),
            torch.tensor([[[0.8]]]),
        ),
        (
            NegativeBinomialLoss(),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[2.0]]]),
            torch.tensor([[[4.0]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[8 / 9]]]),
        ),
    ],
)
def test_scale_params(loss, loc, scale, weights, expected_scaled_loc, expected_scaled_scale):
    scaled_loc, scaled_scale = loss.scale_params(loc, scale, weights)
    torch.testing.assert_close(scaled_loc, expected_scaled_loc, atol=1e-10, rtol=1e-3)
    torch.testing.assert_close(scaled_scale, expected_scaled_scale, atol=1e-10, rtol=1e-3)
