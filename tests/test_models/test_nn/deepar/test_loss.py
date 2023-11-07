import pytest
import torch
from pytorch_lightning import seed_everything

from etna.models.nn.deepar_new.loss import GaussianLoss
from etna.models.nn.deepar_new.loss import NegativeBinomialLoss


@pytest.mark.parametrize(
    "loss,inputs,n_samples,expected_sample",
    [
        (GaussianLoss(), (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])), 2, torch.tensor([1.5410])),
        (GaussianLoss(), (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])), 1, torch.tensor([0.0])),
        (
            NegativeBinomialLoss(),
            (torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0])),
            2,
            torch.tensor([0.0]),
        ),
        (
            NegativeBinomialLoss(),
            (torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0])),
            1,
            torch.tensor([0.125]),
        ),
    ],
)
def test_loss_sample(loss, inputs, n_samples, expected_sample):
    seed_everything(0)

    loss = loss
    loc, scale, weights = inputs
    real_sample = loss.sample(loc, scale, weights, n_samples)
    torch.testing.assert_close(real_sample, expected_sample, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "loss,inputs,expected_loss",
    [
        (
            GaussianLoss(),
            (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0])),
            torch.tensor(1.4189),
        ),
        (
            NegativeBinomialLoss(),
            (torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0])),
            torch.tensor(2.4142),
        ),
    ],
)
def test_loss_forward(loss, inputs, expected_loss):
    seed_everything(0)

    loss = loss
    loc, scale, weights, target = inputs
    real_loss = loss(target, loc, scale, weights)
    torch.testing.assert_close(real_loss, expected_loss, atol=1e-4, rtol=1e-4)
