from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from lightning.pytorch import seed_everything

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models.nn import TFTModel
from etna.models.nn.tft.tft import TFTNet
from etna.pipeline import Pipeline
from etna.transforms import LabelEncoderTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


def ts_label_encode(ts: TSDataset) -> TSDataset:
    seg = SegmentEncoderTransform()
    label1 = LabelEncoderTransform(in_column="categ_exog", out_column="categ_exog_label", strategy="none")
    label2 = LabelEncoderTransform(in_column="categ_regr", out_column="categ_regr_label", strategy="none")
    label3 = LabelEncoderTransform(in_column="categ_regr_new", out_column="categ_regr_new_label", strategy="none")

    ts.fit_transform([seg, label1, label2, label3])

    return ts


@pytest.mark.parametrize(
    "horizon,transform,epochs,lr,eps",
    [
        (8, [StandardScalerTransform(in_column="target")], 30, 0.005, 0.15),
        (14, [StandardScalerTransform(in_column="target")], 35, 0.005, 0.15),
        (16, [StandardScalerTransform(in_column="target")], 35, 0.005, 0.15),
    ],
)
def test_tft_model_run_weekly_overfit(
    random_seed, ts_dataset_weekly_function_with_horizon, horizon, transform, epochs, lr, eps
):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    encoder_length = 14
    decoder_length = 14
    model = TFTModel(
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
    "time_varying_reals_decoder,time_varying_categoricals_decoder,"
    "num_embeddings,features_to_encode",
    [
        ([], [], ["target"], [], [], [], {}, []),
        (
            ["reals_static"],
            ["segment_code"],
            ["reals_exog", "target"],
            ["categ_exog_label", "categ_regr_label", "categ_regr_new_label"],
            ["reals_regr"],
            ["categ_regr_label", "categ_regr_new_label"],
            {"segment_code": 1, "categ_exog_label": 2, "categ_regr_label": 2, "categ_regr_new_label": 1},
            ["categ_exog", "categ_regr", "categ_regr_new"],
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
    num_embeddings,
    features_to_encode,
):
    seed_everything(0, workers=True)
    encoder_length = 4
    decoder_length = 1
    model = TFTModel(
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        static_reals=static_reals,
        static_categoricals=static_categoricals,
        time_varying_reals_encoder=time_varying_reals_encoder,
        time_varying_reals_decoder=time_varying_reals_decoder,
        time_varying_categoricals_encoder=time_varying_categoricals_encoder,
        time_varying_categoricals_decoder=time_varying_categoricals_decoder,
        num_embeddings=num_embeddings,
        trainer_params=dict(max_epochs=1),
    )
    pipeline = Pipeline(
        model=model,
        transforms=[SegmentEncoderTransform()]
        + [
            LabelEncoderTransform(in_column=feature, strategy="none", out_column=f"{feature}_label")
            for feature in features_to_encode
        ],
        horizon=1,
    )
    pipeline.backtest(ts_different_regressors, metrics=[MAE()], n_folds=2)


@pytest.mark.parametrize("ts", ["ts_different_regressors", "ts_different_regressors_int_timestamp"])
@pytest.mark.parametrize(
    "static_reals,static_categoricals,"
    "time_varying_reals_encoder,time_varying_categoricals_encoder,"
    "time_varying_reals_decoder,time_varying_categoricals_decoder,"
    "num_embeddings,features_to_encode",
    [
        ([], [], ["target"], [], [], [], {}, []),
        (
            ["reals_static"],
            ["segment_code"],
            ["reals_exog", "target"],
            ["categ_exog_label", "categ_regr_label", "categ_regr_new_label"],
            ["reals_regr"],
            ["categ_regr_label", "categ_regr_new_label"],
            {"segment_code": 1, "categ_exog_label": 2, "categ_regr_label": 2, "categ_regr_new_label": 1},
            ["categ_exog", "categ_regr_label", "categ_regr_new_label"],
        ),
    ],
)
def test_tft_make_samples(
    ts,
    static_reals,
    static_categoricals,
    time_varying_reals_encoder,
    time_varying_reals_decoder,
    time_varying_categoricals_encoder,
    time_varying_categoricals_decoder,
    num_embeddings,
    features_to_encode,
    request,
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
        num_embeddings=num_embeddings,
    )
    ts = request.getfixturevalue(ts)
    ts_encoded = ts_label_encode(ts)
    df = ts_encoded.to_pandas(flatten=True)
    ts_samples = list(
        TFTNet.make_samples(
            tft_module,
            df=df,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
        )
    )

    first_sample = ts_samples[0]
    assert len(ts_samples) == len(df) - encoder_length - decoder_length + 1
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
        assert first_sample["static_reals"][feature].base is not None
    for feature in static_categoricals:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:1],
            first_sample["static_categoricals"][feature],
        )
        assert first_sample["static_categoricals"][feature].base is not None
    for feature in time_varying_categoricals_encoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:encoder_length],
            first_sample["time_varying_categoricals_encoder"][feature],
        )
        assert first_sample["time_varying_categoricals_encoder"][feature].base is not None
    for feature in time_varying_categoricals_decoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[encoder_length : encoder_length + decoder_length],
            first_sample["time_varying_categoricals_decoder"][feature],
        )
        assert first_sample["time_varying_categoricals_decoder"][feature].base is not None
    for feature in time_varying_reals_encoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[:encoder_length],
            first_sample["time_varying_reals_encoder"][feature],
        )
        assert first_sample["time_varying_reals_encoder"][feature].base is not None
    for feature in time_varying_reals_decoder:
        np.testing.assert_almost_equal(
            df[[feature]].iloc[encoder_length : encoder_length + decoder_length],
            first_sample["time_varying_reals_decoder"][feature],
        )
        assert first_sample["time_varying_reals_decoder"][feature].base is not None


@pytest.mark.parametrize("encoder_length, decoder_length", [(2, 1), (1, 2), (10, 5)])
def test_context_size(encoder_length, decoder_length):
    model = TFTModel(encoder_length=encoder_length, decoder_length=decoder_length)

    assert model.context_size == encoder_length


@pytest.mark.parametrize(
    "static_reals,static_categoricals,"
    "time_varying_reals_encoder,time_varying_categoricals_encoder,"
    "time_varying_reals_decoder,time_varying_categoricals_decoder,"
    "num_embeddings,features_to_encode",
    [
        ([], [], ["target"], [], [], [], {}, []),
        (
            ["reals_static"],
            ["segment_code"],
            ["reals_exog", "target"],
            ["categ_exog_label", "categ_regr_label", "categ_regr_new_label"],
            ["reals_regr"],
            ["categ_regr_label", "categ_regr_new_label"],
            {"segment_code": 1, "categ_exog_label": 2, "categ_regr_label": 2, "categ_regr_new_label": 1},
            ["categ_exog", "categ_regr", "categ_regr_new"],
        ),
    ],
)
def test_save_load(
    ts_different_regressors,
    static_reals,
    static_categoricals,
    time_varying_reals_encoder,
    time_varying_reals_decoder,
    time_varying_categoricals_encoder,
    time_varying_categoricals_decoder,
    num_embeddings,
    features_to_encode,
):
    model = TFTModel(
        encoder_length=3,
        decoder_length=3,
        static_reals=static_reals,
        static_categoricals=static_categoricals,
        time_varying_reals_encoder=time_varying_reals_encoder,
        time_varying_reals_decoder=time_varying_reals_decoder,
        time_varying_categoricals_encoder=time_varying_categoricals_encoder,
        time_varying_categoricals_decoder=time_varying_categoricals_decoder,
        num_embeddings=num_embeddings,
        trainer_params=dict(max_epochs=1),
    )
    assert_model_equals_loaded_original(
        model=model,
        ts=ts_different_regressors,
        transforms=[SegmentEncoderTransform()]
        + [
            LabelEncoderTransform(in_column=feature, strategy="none", out_column=f"{feature}_label")
            for feature in features_to_encode
        ],
        horizon=3,
    )


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = TFTModel(
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
    )
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)


@patch("torch.mps.is_available", return_value=True)
def test_error_training_with_mps(mock_is_available, ts_dataset_weekly_function_with_horizon):
    encoder_length = 14
    decoder_length = 14
    with pytest.raises(NotImplementedError, match="TFTModel does not support MPS. Please use CPU on your MacBook."):
        model = TFTModel(
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            trainer_params=dict(max_epochs=1, accelerator="mps"),
        )


@patch("torch.mps.is_available", return_value=True)
def test_accelerator_cpu_when_mps_is_available(mock_is_available):
    encoder_length = 14
    decoder_length = 14
    model = TFTModel(
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            trainer_params=dict(max_epochs=1),
        )
    assert model.trainer_params["accelerator"] == "cpu"
