from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from etna.models.nn.nbeats.metrics import NBeatsMSE
from etna.models.nn.nbeats.nets import NBeatsBaseNet
from etna.models.nn.nbeats.nets import NBeatsGenericNet
from etna.models.nn.nbeats.nets import NBeatsInterpretableNet


@pytest.fixture
def nbeats_generic_net():
    return NBeatsGenericNet(input_size=5, output_size=3, loss=NBeatsMSE(), stacks=1, layers=1, layer_size=8, lr=0.001)


@pytest.fixture
def nbeats_interpretable_net():
    return NBeatsInterpretableNet(
        input_size=5,
        output_size=3,
        loss=NBeatsMSE(),
        trend_blocks=1,
        trend_layers=1,
        trend_layer_size=8,
        degree_of_polynomial=2,
        seasonality_blocks=1,
        seasonality_layers=1,
        seasonality_layer_size=8,
        lr=0.001,
        num_of_harmonics=1,
    )


@pytest.mark.parametrize("df_name", ["example_make_samples_df", "example_make_samples_df_int_timestamp"])
def test_make_samples(df_name, request):
    df = request.getfixturevalue(df_name)
    module = MagicMock()

    ts_samples = list(NBeatsBaseNet.make_samples(module, df=df, encoder_length=-1, decoder_length=-1))
    first_sample = ts_samples[0]

    assert len(ts_samples) == 1

    expected_first_sample = {
        "history": df["target"].values,
    }

    assert first_sample.keys() == {"history", "history_mask", "target", "target_mask", "segment"}
    assert first_sample["history_mask"] is None
    assert first_sample["target"] is None
    assert first_sample["target_mask"] is None
    assert first_sample["segment"] == "segment_1"
    np.testing.assert_equal(first_sample["history"], expected_first_sample["history"])
    assert first_sample["history"].base is not None


@pytest.mark.parametrize(
    "batch",
    (
        {
            "history": torch.Tensor([[0, 1, 2, 3, 4]]),
            "history_mask": torch.Tensor([[0, 1, 1, 1, 1]]),
            "target": torch.Tensor([[4, 5, 6]]),
            "target_mask": torch.Tensor([[1, 1, 1]]),
        },
    ),
)
@pytest.mark.parametrize(
    "net_name",
    (
        "nbeats_generic_net",
        "nbeats_interpretable_net",
    ),
)
def test_step(net_name, batch, request):
    net = request.getfixturevalue(net_name)
    loss, target, forecast = net.step(batch)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(forecast, torch.Tensor)
    assert torch.all(target == batch["target"])
    assert forecast.shape == batch["target"].shape


@pytest.mark.parametrize(
    "net_name",
    ("nbeats_generic_net", "nbeats_interpretable_net"),
)
def test_configure_optimizer(net_name, request):
    net = request.getfixturevalue(net_name)
    optimizers, schedulers = net.configure_optimizers()

    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(schedulers[0]["scheduler"], torch.optim.lr_scheduler.LambdaLR)
