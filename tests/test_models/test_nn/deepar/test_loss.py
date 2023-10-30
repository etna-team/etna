import pytest
import torch
from pytorch_lightning import seed_everything

from etna.models.nn.deepar_new.loss import GaussianLoss
from etna.models.nn.deepar_new.loss import NegativeBinomialLoss


@pytest.mark.parametrize(
    "loss,inputs,n_samples,expected_ans",
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
            (torch.tensor([[2.0]]), torch.tensor([[2.0]]), torch.tensor([[1.0]])),
            1,
            torch.tensor([0.125]),
        ),
    ],
)
def test_loss_sample(loss, inputs, n_samples, expected_ans):
    seed_everything(0)

    loss = loss
    loc, scale, weights = inputs
    ans = loss.sample(loc, scale, weights, n_samples)
    assert ans == expected_ans  # TODO first test fails


@pytest.mark.parametrize(
    "loss,inputs,expected_ans",
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
def test_loss_forward(loss, inputs, expected_ans):
    seed_everything(0)

    loss = loss
    loc, scale, weights, target = inputs
    ans = loss(target, loc, scale, weights)
    assert ans == expected_ans  # TODO test fail
