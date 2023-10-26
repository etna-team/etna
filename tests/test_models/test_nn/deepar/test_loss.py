import random

import numpy as np
import torch
from torch.distributions import NegativeBinomial
from torch.distributions import Normal

from etna.models.nn.deepar_new import GaussianLoss
from etna.models.nn.deepar_new import NegativeBinomialLoss


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def test_gaussian_loss():
    seed_everything()
    true_values = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    std = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    loss = GaussianLoss()
    sample = loss.sample(mean, std)
    loss_value = loss(inputs=true_values, mean=mean, std=std**2)

    torch_loss = Normal(loc=mean, scale=std)
    torch_sample = torch_loss.sample()
    torch_loss_value = -torch_loss.log_prob(true_values).sum()

    print(sample, torch_sample)
    print(loss_value, torch_loss_value)
    assert sample == torch_sample
    assert loss_value == torch_loss_value
