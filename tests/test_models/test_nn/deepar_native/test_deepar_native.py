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


@pytest.mark.parametrize("df_name", ["example_make_samples_df", "example_make_samples_df_int_timestamp"])
@pytest.mark.parametrize("scale, weight_1, weight_2", [(False, 1, 1), (True, 3, 4)])
def test_deepar_make_samples(df_name, scale, weight_1, weight_2, request):
    df = request.getfixturevalue(df_name)
    deepar_module = MagicMock(scale=scale)
    encoder_length = 4
    decoder_length = 1

    ts_samples = list(
        DeepARNativeNet.make_samples(
            deepar_module,
            df=df,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
        )
    )
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]

    assert len(ts_samples) == len(df) - encoder_length - decoder_length + 1

    df["target_shifted"] = df["target"].shift(1)
    df["target_shifted_scaled_1"] = df["target_shifted"] / weight_1
    df["target_shifted_scaled_2"] = df["target_shifted"] / weight_2
    expected_first_sample = {
        "encoder_real": df[["target_shifted_scaled_1", "regressor_float", "regressor_int"]]
        .iloc[1:encoder_length]
        .values,
        "decoder_real": df[["target_shifted_scaled_1", "regressor_float", "regressor_int"]]
        .iloc[encoder_length : encoder_length + decoder_length]
        .values,
        "encoder_target": df[["target"]].iloc[1:encoder_length].values,
        "decoder_target": df[["target"]].iloc[encoder_length : encoder_length + decoder_length].values,
        "weight": weight_1,
    }
    expected_second_sample = {
        "encoder_real": df[["target_shifted_scaled_2", "regressor_float", "regressor_int"]]
        .iloc[2 : encoder_length + 1]
        .values,
        "decoder_real": df[["target_shifted_scaled_2", "regressor_float", "regressor_int"]]
        .iloc[encoder_length + 1 : encoder_length + decoder_length + 1]
        .values,
        "encoder_target": df[["target"]].iloc[2 : encoder_length + 1].values,
        "decoder_target": df[["target"]].iloc[encoder_length + 1 : encoder_length + decoder_length + 1].values,
        "weight": weight_2,
    }

    assert first_sample.keys() == {
        "encoder_real",
        "decoder_real",
        "encoder_target",
        "decoder_target",
        "segment",
        "weight",
    }
    assert first_sample["segment"] == "segment_1"
    for key in expected_first_sample:
        np.testing.assert_equal(first_sample[key], expected_first_sample[key])

    assert second_sample.keys() == {
        "encoder_real",
        "decoder_real",
        "encoder_target",
        "decoder_target",
        "segment",
        "weight",
    }
    assert second_sample["segment"] == "segment_1"
    for key in expected_second_sample:
        np.testing.assert_equal(second_sample[key], expected_second_sample[key])


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
