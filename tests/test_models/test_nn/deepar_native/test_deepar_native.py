from unittest.mock import MagicMock

import numpy as np
import pytest
from lightning.pytorch import seed_everything

from etna.metrics import MAE
from etna.models.nn import DeepARModel
from etna.models.nn.deepar.deepar import DeepARNet
from etna.models.nn.deepar.loss import GaussianLoss
from etna.models.nn.deepar.loss import NegativeBinomialLoss
from etna.pipeline import Pipeline
from etna.transforms import FilterFeaturesTransform
from etna.transforms import LabelEncoderTransform
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
        (13, NegativeBinomialLoss(), [], 900, 1e-2, 0.06),
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
    model = DeepARModel(
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


@pytest.mark.parametrize(
    "embedding_sizes,features_to_encode",
    [({}, []), ({"categ_regr_label": (2, 5), "categ_regr_new_label": (1, 5)}, ["categ_regr", "categ_regr_new"])],
)
def test_handling_categoricals(ts_different_regressors, embedding_sizes, features_to_encode):
    encoder_length = 4
    decoder_length = 1
    model = DeepARModel(
        input_size=3,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        embedding_sizes=embedding_sizes,
        trainer_params=dict(max_epochs=1),
    )
    pipeline = Pipeline(
        model=model,
        transforms=[
            LabelEncoderTransform(in_column=feature, strategy="none", out_column=f"{feature}_label")
            for feature in features_to_encode
        ]
        + [FilterFeaturesTransform(exclude=["reals_exog", "categ_exog"])],
        horizon=1,
    )
    pipeline.backtest(ts_different_regressors, metrics=[MAE()], n_folds=2)


@pytest.mark.parametrize("cat_columns", [[], ["regressor_int_cat"]])
@pytest.mark.parametrize("df_name", ["example_make_samples_df", "example_make_samples_df_int_timestamp"])
@pytest.mark.parametrize("scale, weights", [(False, [1, 1]), (True, [3, 4])])
def test_deepar_make_samples(df_name, scale, weights, cat_columns, request):
    df = request.getfixturevalue(df_name)
    deepar_module = MagicMock(scale=scale, embedding_sizes={column: (7, 1) for column in cat_columns})
    encoder_length = 4
    decoder_length = 1

    ts_samples = list(
        DeepARNet.make_samples(
            deepar_module,
            df=df,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
        )
    )

    assert len(ts_samples) == len(df) - encoder_length - decoder_length + 1

    num_samples_check = 2
    df["target_shifted"] = df["target"].shift(1)
    for i in range(num_samples_check):
        expected_sample = {
            "encoder_real": df[["target_shifted", "regressor_float", "regressor_int"]]
            .iloc[1 + i : encoder_length + i]
            .values,
            "decoder_real": df[["target_shifted", "regressor_float", "regressor_int"]]
            .iloc[encoder_length + i : encoder_length + decoder_length + i]
            .values,
            "encoder_categorical": {
                column: df[[column]].iloc[1 + i : encoder_length + i].values for column in cat_columns
            },
            "decoder_categorical": {
                column: df[[column]].iloc[encoder_length + i : encoder_length + decoder_length + i].values
                for column in cat_columns
            },
            "encoder_target": df[["target"]].iloc[1 + i : encoder_length + i].values,
            "decoder_target": df[["target"]].iloc[encoder_length + i : encoder_length + decoder_length + i].values,
            "weight": weights[i],
        }

        assert ts_samples[i].keys() == {
            "encoder_real",
            "decoder_real",
            "encoder_categorical",
            "decoder_categorical",
            "encoder_target",
            "decoder_target",
            "segment",
            "weight",
        }
        assert ts_samples[i]["segment"] == "segment_1"
        for key in expected_sample:
            np.testing.assert_equal(ts_samples[i][key], expected_sample[key])
            if "categorical" in key:
                for column in ts_samples[i][key]:
                    assert ts_samples[i][key][column].base is not None
            else:
                if key != "weight":
                    assert ts_samples[i][key].base is not None


@pytest.mark.parametrize("encoder_length", [1, 2, 10])
def test_context_size(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    model = DeepARModel(
        input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=100)
    )

    assert model.context_size == encoder_length


def test_save_load(example_tsds):
    model = DeepARModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = DeepARModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
