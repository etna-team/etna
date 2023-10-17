import torch

from etna.models.nn.deepar_new import DeepARModelNew, DeepARNetNew
from torch.distributions import Normal, NegativeBinomial
from torch import nn
import pytest


@pytest.mark.parametrize("loss,true_params", [(Normal, (torch.tensor(1.), torch.tensor(1.3132))), (NegativeBinomial, (torch.tensor(0.7614), torch.tensor(0.3670)))])
def test_count_distr_params(loss, true_params):
    net = DeepARNetNew(num_layers=1,
                       hidden_size=2,
                       input_size=4, lr=1e-3, loss=loss, optimizer_params={})
    net.loc = nn.Linear(2, 1, False)
    net.loc.weight = torch.nn.Parameter(torch.tensor([3., -2.]))
    net.scale = nn.Linear(2, 1, False)
    net.scale.weight = torch.nn.Parameter(torch.tensor([3., -2.]))
    distr = net._count_distr_params(torch.Tensor([1., 1.]), 1)
    if isinstance(distr, Normal):
        torch.testing.assert_close(distr.loc, true_params[0], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(distr.scale, true_params[1], rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(distr.total_count, true_params[0], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(distr.probs, true_params[1], rtol=1e-3, atol=1e-3)
