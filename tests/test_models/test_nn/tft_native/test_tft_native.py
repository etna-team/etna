import numpy as np
import pytest
from pytorch_lightning import seed_everything

from etna.metrics import MAE
from etna.models.nn import TFTNativeModel
from etna.models.nn.tft_native.tft import TFTNativeNet
from etna.pipeline import Pipeline
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.parametrize(
    "horizon,transform,epochs,lr,eps",
    [
        (8, [StandardScalerTransform(in_column="target")], 30, 0.005, 0.05),
        (13, [StandardScalerTransform(in_column="target")], 35, 0.005, 0.06),
    ],
)
def test_tft_model_run_weekly_overfit(ts_dataset_weekly_function_with_horizon, horizon, transform, epochs, lr, eps):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """
    seed_everything(0, workers=True)
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    encoder_length = 14
    decoder_length = 14
    model = TFTNativeModel(
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        lr=lr,
        trainer_params=dict(max_epochs=epochs),
    )
    pipeline = Pipeline(model=model, transforms=transform, horizon=horizon)
    pipeline.fit(ts_train)
    future = pipeline.forecast()

    mae = MAE("macro")
    assert mae(ts_test, future) < eps


def test_tft_make_samples(df_with_ascending_window_mean):
    encoder_length = 5
    decoder_length = 1
    tft_module = TFTNativeModel(encoder_length=encoder_length, decoder_length=decoder_length)

    ts_samples = list(
        TFTNativeNet.make_samples(
            tft_module,
            df=df_with_ascending_window_mean,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
        )
    )
    first_sample = ts_samples[0]

    assert first_sample["segment"] == "segment_1"
    assert first_sample["decoder_target"].shape == (decoder_length, 1)

    assert len(first_sample["static_reals"]) == 0
    assert len(first_sample["static_categoricals"]) == 0
    assert len(first_sample["time_varying_categoricals_encoder"]) == 0
    assert len(first_sample["time_varying_categoricals_decoder"]) == 0
    assert len(first_sample["time_varying_reals_encoder"]) == 1
    assert len(first_sample["time_varying_reals_decoder"]) == 0

    np.testing.assert_almost_equal(
        df_with_ascending_window_mean[["target"]].iloc[:encoder_length],
        first_sample["time_varying_reals_encoder"]["target"],
    )


@pytest.mark.parametrize("encoder_length", [1, 2, 10])
def test_context_size(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    model = TFTNativeModel(
        encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=100)
    )

    assert model.context_size == encoder_length


def test_save_load(example_tsds):
    model = TFTNativeModel(encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = TFTNativeModel(encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
