from unittest.mock import MagicMock

import numpy as np
import pytest
from pytorch_lightning import seed_everything

from etna.metrics import MAE
from etna.models.nn import DeepARNativeModel
from etna.models.nn.deepar_native.deepar import DeepARNativeNet
from etna.models.nn.deepar_native.loss import GaussianLoss
from etna.models.nn.deepar_native.loss import NegativeBinomialLoss
from etna.pipeline import Pipeline
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.parametrize(
    "horizon,loss,transform,epochs,lr,eps",
    [
        (8, GaussianLoss(), [StandardScalerTransform(in_column="target")], 100, 1e-3, 0.05),
        (13, GaussianLoss(), [StandardScalerTransform(in_column="target")], 100, 1e-3, 0.05),
        (15, GaussianLoss(), [StandardScalerTransform(in_column="target")], 100, 1e-3, 0.05),
        (8, NegativeBinomialLoss(), [], 300, 1e-2, 0.05),
        (13, NegativeBinomialLoss(), [], 300, 1e-2, 0.06),
        (15, NegativeBinomialLoss(), [], 300, 1e-2, 0.05),
    ],
)
def test_deepar_model_run_weekly_overfit(
    ts_dataset_weekly_function_with_horizon, horizon, loss, transform, epochs, lr, eps
):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """
    seed_everything(0, workers=True)
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    encoder_length = 14
    decoder_length = 14
    model = DeepARNativeModel(
        input_size=1,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        scale=False,
        lr=lr,
        trainer_params=dict(max_epochs=epochs),
        loss=loss,
    )
    pipeline = Pipeline(model=model, transforms=transform, horizon=horizon)
    pipeline.fit(ts_train)
    future = pipeline.forecast()

    mae = MAE("macro")
    assert mae(ts_test, future) < eps


@pytest.mark.parametrize("scale, mean_1, mean_2", [(False, 0, 0), (True, 3, 4)])
def test_deepar_make_samples(df_with_ascending_window_mean, scale, mean_1, mean_2):
    deepar_module = MagicMock(scale=scale)
    encoder_length = 4
    decoder_length = 1

    ts_samples = list(
        DeepARNativeNet.make_samples(
            deepar_module,
            df=df_with_ascending_window_mean,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
        )
    )
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]

    assert first_sample["segment"] == "segment_1"
    assert first_sample["encoder_real"].shape == (encoder_length - 1, 1)
    assert first_sample["decoder_real"].shape == (decoder_length, 1)
    assert first_sample["encoder_target"].shape == (encoder_length - 1, 1)
    assert first_sample["decoder_target"].shape == (decoder_length, 1)
    np.testing.assert_almost_equal(
        df_with_ascending_window_mean[["target"]].iloc[: encoder_length - 1],
        first_sample["encoder_real"] * (1 + mean_1),
    )
    np.testing.assert_almost_equal(
        df_with_ascending_window_mean[["target"]].iloc[1:encoder_length], second_sample["encoder_real"] * (1 + mean_2)
    )


@pytest.mark.parametrize("encoder_length", [1, 2, 10])
def test_context_size(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    model = DeepARNativeModel(
        input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=100)
    )

    assert model.context_size == encoder_length


def test_save_load(example_tsds):
    model = DeepARNativeModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = DeepARNativeModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
