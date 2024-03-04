from unittest.mock import MagicMock

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
        (13, [StandardScalerTransform(in_column="target")], 35, 0.005, 0.08),
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


@pytest.mark.parametrize(
    "static_reals,static_categoricals,"
    "time_varying_reals_encoder,time_varying_categoricals_encoder,"
    "time_varying_reals_decoder,time_varying_categoricals_decoder",
    [
        ([], [], ["target"], [], [], []),
        (
            ["cont_static"],
            ["segment"],
            ["cont", "target"],
            ["categ", "categ_exog", "categ_exog_new"],
            ["cont_exog"],
            ["categ_exog", "categ_exog_new"],
        ),
    ],
)
def test_tft_backtest(
    ts_different_regressors,
    static_reals,
    static_categoricals,
    time_varying_reals_encoder,
    time_varying_reals_decoder,
    time_varying_categoricals_encoder,
    time_varying_categoricals_decoder,
):
    seed_everything(0, workers=True)
    encoder_length = 4
    decoder_length = 1
    model = TFTNativeModel(
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        static_reals=static_reals,
        static_categoricals=static_categoricals,
        time_varying_reals_encoder=time_varying_reals_encoder,
        time_varying_reals_decoder=time_varying_reals_decoder,
        time_varying_categoricals_encoder=time_varying_categoricals_encoder,
        time_varying_categoricals_decoder=time_varying_categoricals_decoder,
        trainer_params=dict(max_epochs=1),
    )
    pipeline = Pipeline(model=model, transforms=[], horizon=1)
    pipeline.backtest(ts_different_regressors, metrics=[MAE()], n_folds=2)


@pytest.mark.parametrize(
    "static_reals,static_categoricals,"
    "time_varying_reals_encoder,time_varying_categoricals_encoder,"
    "time_varying_reals_decoder,time_varying_categoricals_decoder,"
    "categorical_feature_to_id",
    [
        ([], [], ["target"], [], [], [], {}),
        (
            ["cont_static"],
            ["segment"],
            ["cont", "target"],
            ["categ", "categ_exog", "categ_exog_new"],
            ["cont_exog"],
            ["categ_exog", "categ_exog_new"],
            {
                "segment": {"segment_0": 0},
                "categ": {"a": 0, "d": 1},
                "categ_exog": {"b": 0, "e": 1},
                "categ_exog_new": {"b": 0},
            },
        ),
    ],
)
def test_tft_make_samples(
    ts_different_regressors,
    static_reals,
    static_categoricals,
    time_varying_reals_encoder,
    time_varying_reals_decoder,
    time_varying_categoricals_encoder,
    time_varying_categoricals_decoder,
    categorical_feature_to_id
):
    encoder_length = 4
    decoder_length = 1
    tft_module = MagicMock(
        static_reals=static_reals,
        static_categoricals=static_categoricals,
        time_varying_reals_encoder=time_varying_reals_encoder,
        time_varying_reals_decoder=time_varying_reals_decoder,
        time_varying_categoricals_encoder=time_varying_categoricals_encoder,
        time_varying_categoricals_decoder=time_varying_categoricals_decoder,
        categorical_feature_to_id=categorical_feature_to_id,
    )
    df = ts_different_regressors.to_pandas(flatten=True)
    ts_samples = list(
        TFTNativeNet.make_samples(
            tft_module,
            df=df,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
        )
    )
    first_sample = ts_samples[0]

    assert first_sample["segment"] == "segment_0"
    assert first_sample["decoder_target"].shape == (decoder_length, 1)

    assert len(first_sample["static_reals"]) == len(static_reals)
    assert len(first_sample["static_categoricals"]) == len(static_categoricals)
    assert len(first_sample["time_varying_categoricals_encoder"]) == len(time_varying_categoricals_encoder)
    assert len(first_sample["time_varying_categoricals_decoder"]) == len(time_varying_categoricals_decoder)
    assert len(first_sample["time_varying_reals_encoder"]) == len(time_varying_reals_encoder)
    assert len(first_sample["time_varying_reals_decoder"]) == len(time_varying_reals_decoder)

    for feature in static_reals:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:1],
            first_sample["static_reals"][feature],
        )
    for feature in static_categoricals:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:1],
            first_sample["static_categoricals"][feature],
        )
    for feature in time_varying_categoricals_encoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:encoder_length],
            first_sample["time_varying_categoricals_encoder"][feature],
        )
    for feature in time_varying_categoricals_decoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[encoder_length : encoder_length + decoder_length],
            first_sample["time_varying_categoricals_decoder"][feature],
        )
    for feature in time_varying_reals_encoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:encoder_length],
            first_sample["time_varying_reals_encoder"][feature],
        )
    for feature in time_varying_reals_decoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[encoder_length : encoder_length + decoder_length],
            first_sample["time_varying_reals_decoder"][feature],
        )


@pytest.mark.parametrize("encoder_length", [1, 2, 10])
def test_context_size(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    model = TFTNativeModel(encoder_length=encoder_length, decoder_length=decoder_length)

    assert model.context_size == encoder_length


def test_save_load(example_tsds):
    model = TFTNativeModel(encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_save_load_new(ts_different_regressors):
    model = TFTNativeModel(
        encoder_length=3,
        decoder_length=3,
        static_categoricals=['segment'],
        time_varying_reals_encoder=['cont', 'cont_exog'],
        time_varying_reals_decoder=['cont_exog'],
        time_varying_categoricals_encoder=['categ', 'categ_exog', 'categ_exog_new'],
        time_varying_categoricals_decoder=['categ_exog', 'categ_exog_new'],
        trainer_params=dict(max_epochs=1)
    )
    assert_model_equals_loaded_original(model=model, ts=ts_different_regressors, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = TFTNativeModel(encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
