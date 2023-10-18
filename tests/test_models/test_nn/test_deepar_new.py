import torch

from etna.models.nn.deepar_new import DeepARNetNew
from etna.models.nn import DeepARModelNew
from torch.distributions import Normal, NegativeBinomial
from torch import nn
import pytest
from unittest.mock import MagicMock
from etna.transforms import StandardScalerTransform

import numpy as np

from etna.metrics import MAE
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15,
    ],
)
def test_deepar_model_run_weekly_overfit(ts_dataset_weekly_function_with_horizon, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    encoder_length = 14
    decoder_length = 14
    model = DeepARModelNew(
        input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=10)
    )
    future = ts_train.make_future(horizon, tail_steps=encoder_length)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)

    mae = MAE("macro")
    assert mae(ts_test, future) < 3  # TODO fix, in rnn test mae is lower


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


def test_params_to_tune(example_tsds):  # TODO
    ts = example_tsds
    model = DeepARModelNew(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
