from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import NegativeBinomial
from torch.distributions import Normal

from etna.metrics import MAE
from etna.models.nn import DeepARModelNew
from etna.models.nn.deepar_new.deepar import DeepARNetNew
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
    assert mae(ts_test, future) < 6  # TODO fix, in rnn test mae is lower


def test_deepar_make_samples(example_df):
    deepar_module = MagicMock()
    encoder_length = 8
    decoder_length = 4

    mean = example_df["target"].mean()
    print(mean)
    print(example_df)
    df_copied = example_df.copy(deep=True)
    ts_samples = list(
        DeepARNetNew.make_samples(
            deepar_module, df=example_df, encoder_length=encoder_length, decoder_length=decoder_length
        )
    )
    zero_sample = ts_samples[0]
    first_notzero_sample = ts_samples[encoder_length - 2]
    second_notzero_sample = ts_samples[encoder_length - 1]

    zero_sample_true = np.array([0] * (encoder_length - 2) + df_copied[["target"]].iloc[0].tolist()).reshape(-1, 1)

    assert first_notzero_sample["segment"] == "segment_1"
    assert first_notzero_sample["encoder_real"].shape == (encoder_length - 1, 1)
    assert first_notzero_sample["decoder_real"].shape == (decoder_length, 1)
    assert first_notzero_sample["encoder_target"].shape == (encoder_length - 1, 1)
    assert first_notzero_sample["decoder_target"].shape == (decoder_length, 1)

    np.testing.assert_allclose(zero_sample_true, zero_sample["encoder_real"] * mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        df_copied[["target"]].iloc[: encoder_length - 1],
        first_notzero_sample["encoder_real"] * mean,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        df_copied[["target"]].iloc[1:encoder_length],
        second_notzero_sample["encoder_real"] * mean,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("encoder_length", [1, 2, 10])
def test_context_size(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    model = DeepARModelNew(
        input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=100)
    )

    assert model.context_size == encoder_length


def test_save_load(example_tsds):
    model = DeepARModelNew(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.mark.parametrize(
    "loss,true_params,weight",
    [
        (Normal, (torch.tensor([[2.0]]), torch.tensor([[2.6265]])), torch.tensor([2])),
        (NegativeBinomial, (torch.tensor([[0.7614]]), torch.tensor([[0.3670]])), torch.tensor([1])),
    ],
)
def test_count_distr_params(loss, true_params, weight):
    net = DeepARNetNew(num_layers=1, hidden_size=2, input_size=4, lr=1e-3, loss=loss, optimizer_params={})
    net.loc = nn.Linear(2, 1, False)
    net.loc.weight = torch.nn.Parameter(torch.tensor([[3.0, -2.0]]))
    net.scale = nn.Linear(2, 1, False)
    net.scale.weight = torch.nn.Parameter(torch.tensor([[3.0, -2.0]]))
    distr = net._count_distr_params(torch.Tensor([[1.0, 1.0]]), weight)
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
