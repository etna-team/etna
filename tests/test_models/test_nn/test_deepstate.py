from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import torch

from etna.metrics import MAE
from etna.models.nn import DeepStateModel
from etna.models.nn.deepstate import CompositeSSM
from etna.models.nn.deepstate import WeeklySeasonalitySSM
from etna.models.nn.deepstate.deepstate import DeepStateNet
from etna.pipeline import Pipeline
from etna.transforms import FilterFeaturesTransform
from etna.transforms import LabelEncoderTransform
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.parametrize(
    "horizon",
    [
        1,
        8,
        13,
        15,
    ],
)
def test_deepstate_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When: I use scale transformations
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    std = StandardScalerTransform(in_column="target")
    ts_train.fit_transform([std])

    encoder_length = 14
    decoder_length = 14
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        trainer_params=dict(max_epochs=100),
    )
    future = ts_train.make_future(horizon, transforms=[std], tail_steps=encoder_length)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)
    future.inverse_transform([std])

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.001


@pytest.mark.parametrize(
    "embedding_sizes,features_to_encode",
    [
        ({}, []),
        ({"categ_regr_label": (2, 5), "categ_regr_new_label": (1, 5)}, ["categ_regr", "categ_regr_new"]),
    ],
)
def test_handling_categoricals(ts_different_regressors, embedding_sizes, features_to_encode):
    encoder_length = 2
    decoder_length = 1
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=2,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        embedding_sizes=embedding_sizes,
        trainer_params=dict(max_epochs=100, accelerator="cpu" if torch.mps.is_available() else "auto"),
    )
    pipeline = Pipeline(
        model=model,
        transforms=[
            LabelEncoderTransform(in_column=feature, strategy="none", out_column=f"{feature}_label")
            for feature in features_to_encode
        ]
        + [FilterFeaturesTransform(exclude=["reals_exog", "categ_exog", "categ_regr", "categ_regr_new"])],
        horizon=2,
    )
    pipeline.backtest(ts_different_regressors, metrics=[MAE()], n_folds=2)


@pytest.mark.parametrize("cat_columns", [[], ["regressor_int_cat"]])
@pytest.mark.parametrize("df_name", ["example_make_samples_df", "example_make_samples_df_int_timestamp"])
def test_deepstate_make_samples(df_name, cat_columns, request):
    df = request.getfixturevalue(df_name)

    # this model can't work with this type of columns
    df.drop(columns=["regressor_bool", "regressor_str"], inplace=True)

    ssm = MagicMock()
    datetime_index = np.arange(len(df))
    ssm.generate_datetime_index.return_value = datetime_index[np.newaxis, :]
    module = MagicMock(ssm=ssm, embedding_sizes={column: (7, 1) for column in cat_columns})
    encoder_length = 8
    decoder_length = 4

    ts_samples = list(
        DeepStateNet.make_samples(module, df=df, encoder_length=encoder_length, decoder_length=decoder_length)
    )

    assert len(ts_samples) == len(df) - encoder_length - decoder_length + 1

    num_samples_check = 2
    df["datetime_index"] = datetime_index
    for i in range(num_samples_check):
        expected_sample = {
            "encoder_real": df[["regressor_float", "regressor_int", "regressor_int_cat"]]
            .drop(columns=cat_columns)
            .iloc[i : encoder_length + i]
            .values,
            "decoder_real": df[["regressor_float", "regressor_int", "regressor_int_cat"]]
            .drop(columns=cat_columns)
            .iloc[encoder_length + i : encoder_length + decoder_length + i]
            .values,
            "encoder_categorical": {column: df[[column]].iloc[i : encoder_length + i].values for column in cat_columns},
            "decoder_categorical": {
                column: df[[column]].iloc[encoder_length + i : encoder_length + decoder_length + i].values
                for column in cat_columns
            },
            "encoder_target": df[["target"]].iloc[i : encoder_length + i].values,
            "datetime_index": df[["datetime_index"]].iloc[i : encoder_length + decoder_length + i].values.T,
        }

        assert ts_samples[i].keys() == {
            "encoder_real",
            "decoder_real",
            "encoder_categorical",
            "decoder_categorical",
            "encoder_target",
            "datetime_index",
            "segment",
        }
        assert ts_samples[i]["segment"] == "segment_1"
        for key in expected_sample:
            np.testing.assert_equal(ts_samples[i][key], expected_sample[key])
            if "categorical" in key:
                for column in ts_samples[i][key]:
                    assert ts_samples[i][key][column].base is not None
            else:
                assert ts_samples[i][key].base is not None


def test_save_load(example_tsds):
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=100, accelerator="cpu" if torch.mps.is_available() else "auto"),
    )
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=100, accelerator="cpu" if torch.mps.is_available() else "auto"),
    )
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)


@patch("lightning.pytorch.accelerators.mps.MPSAccelerator.is_available", return_value=True)
def test_error_training_with_mps(mock_is_available, ts_dataset_weekly_function_with_horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(8)

    encoder_length = 14
    decoder_length = 14
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        trainer_params=dict(max_epochs=100),
    )
    with pytest.raises(
        NotImplementedError, match="DeepStateModel does not support MPS. Please use CPU on your MacBook."
    ):
        model.fit(ts_train)
