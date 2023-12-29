from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import MAE
from etna.models.nn import DeepStateModel
from etna.models.nn.deepstate import CompositeSSM
from etna.models.nn.deepstate import WeeklySeasonalitySSM
from etna.models.nn.deepstate.deepstate import DeepStateNet
from etna.transforms import StandardScalerTransform
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


def test_fit_int_timestamp_fail(example_tsds_int_timestamp):
    ts = example_tsds_int_timestamp
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
    )
    with pytest.raises(ValueError, match="Invalid timestamp! Only datetime type is supported."):
        model.fit(ts)


def test_fit_external_timestamp_not_present_fail(example_tsds):
    ts = example_tsds
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
        timestamp_column="unknown_feature",
    )
    with pytest.raises(ValueError, match="Invalid timestamp_column! It isn't present in a given dataset."):
        model.fit(ts)


def test_fit_external_timestamp_not_regressor_fail():
    df = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_wide = TSDataset.to_dataset(df)
    df_exog = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_exog["target"] = pd.date_range(start="2020-01-01", periods=100)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    df_exog_wide.rename(columns={"target": "external_timestamp"}, level="feature", inplace=True)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, known_future=[], freq=None)

    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
        timestamp_column="external_timestamp",
    )
    with pytest.raises(ValueError, match="Invalid timestamp_column! It should be a regressor."):
        model.fit(ts)


def test_fit_external_timestamp_not_datetime_fail():
    df = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_wide = TSDataset.to_dataset(df)
    df_exog = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_exog["target"] = np.arange(100)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    df_exog_wide.rename(columns={"target": "external_timestamp"}, level="feature", inplace=True)
    ts = TSDataset(df=df_wide.iloc[:-5], df_exog=df_exog_wide, known_future="all", freq=None)

    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
        timestamp_column="external_timestamp",
    )
    with pytest.raises(ValueError, match="Invalid timestamp_column! Only datetime type is supported."):
        model.fit(ts)


def test_fit_external_timestamp_not_sequential_fail():
    df = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_wide = TSDataset.to_dataset(df)
    df_exog = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_exog["target"] = (
        pd.date_range(start="2020-01-01", periods=50).tolist() + pd.date_range(start="2021-01-01", periods=50).tolist()
    )
    df_exog_wide = TSDataset.to_dataset(df_exog)
    df_exog_wide.rename(columns={"target": "external_timestamp"}, level="feature", inplace=True)
    ts = TSDataset(df=df_wide.iloc[:-5], df_exog=df_exog_wide, known_future="all", freq=None)

    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
        timestamp_column="external_timestamp",
    )
    with pytest.raises(ValueError, match="Invalid timestamp_column! It doesn't contain sequential timestamps."):
        model.fit(ts)


@pytest.mark.parametrize("timestamp_column", ["timestamp", "external_timestamp"])
@pytest.mark.parametrize("df_name", ["example_make_samples_df", "example_make_samples_df_int_timestamp"])
def test_deepstate_make_samples(timestamp_column, df_name, request):
    df = request.getfixturevalue(df_name)
    # this model can't work with this type of columns
    df.drop(columns=["regressor_bool", "regressor_str"], inplace=True)

    if timestamp_column != "timestamp":
        df["external_timestamp"] = df["timestamp"]

    ssm = MagicMock()
    datetime_index = np.arange(len(df))
    ssm.generate_datetime_index.return_value = datetime_index[np.newaxis, :]
    module = MagicMock(ssm=ssm, timestamp_column=timestamp_column)
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
            .iloc[i : encoder_length + i]
            .values,
            "decoder_real": df[["regressor_float", "regressor_int", "regressor_int_cat"]]
            .iloc[encoder_length + i : encoder_length + decoder_length + i]
            .values,
            "encoder_target": df[["target"]].iloc[i : encoder_length + i].values,
            "datetime_index": df[["datetime_index"]].iloc[i : encoder_length + decoder_length + i].values.T,
        }

        assert ts_samples[i].keys() == {"encoder_real", "decoder_real", "encoder_target", "datetime_index", "segment"}
        assert ts_samples[i]["segment"] == "segment_1"
        for key in expected_sample:
            np.testing.assert_equal(ts_samples[i][key], expected_sample[key])


def test_save_load(example_tsds):
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
    )
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = DeepStateModel(
        ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()], nonseasonal_ssm=None),
        input_size=0,
        encoder_length=14,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
    )
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
