import math
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from etna.metrics import MAE
from etna.models.nn import MLPModel
from etna.models.nn.mlp import MLPNet
from etna.pipeline import Pipeline
from etna.transforms import FilterFeaturesTransform
from etna.transforms import FourierTransform
from etna.transforms import LabelEncoderTransform
from etna.transforms import LagTransform
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
def test_mlp_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 4)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    transforms = [std, lag, fourier]
    ts_train.fit_transform(transforms)

    decoder_length = 14
    model = MLPModel(
        input_size=10,
        hidden_size=[10, 10, 10, 10, 10],
        lr=1e-1,
        decoder_length=decoder_length,
        trainer_params=dict(max_epochs=100),
    )
    future = ts_train.make_future(horizon, transforms=transforms)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)
    future.inverse_transform(transforms)

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.05


@pytest.mark.parametrize(
    "embedding_sizes,features_to_encode",
    [({}, []), ({"categ_regr_label": (2, 5), "categ_regr_new_label": (1, 5)}, ["categ_regr", "categ_regr_new"])],
)
def test_handling_categoricals(ts_different_regressors, embedding_sizes, features_to_encode):
    decoder_length = 4
    model = MLPModel(
        input_size=2,
        hidden_size=[10],
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
def test_mlp_make_samples(df_name, cat_columns, request):
    df = request.getfixturevalue(df_name)
    mlp_module = MagicMock(embedding_sizes={column: (7, 1) for column in cat_columns})
    encoder_length = 0
    decoder_length = 5
    ts_samples = list(
        MLPNet.make_samples(mlp_module, df=df, encoder_length=encoder_length, decoder_length=decoder_length)
    )

    assert len(ts_samples) == math.ceil(len(df) / decoder_length)

    num_samples_check = 2
    for i in range(num_samples_check):
        expected_sample = {
            "decoder_real": df[["regressor_float", "regressor_int"]]
            .iloc[encoder_length + decoder_length * i : encoder_length + decoder_length * (i + 1)]
            .values,
            "decoder_categorical": {
                column: df[[column]]
                .iloc[encoder_length + decoder_length * i : encoder_length + decoder_length * (i + 1)]
                .values
                for column in cat_columns
            },
            "decoder_target": df[["target"]]
            .iloc[encoder_length + decoder_length * i : encoder_length + decoder_length * (i + 1)]
            .values,
        }

        assert ts_samples[i].keys() == {"decoder_real", "decoder_categorical", "decoder_target", "segment"}
        assert ts_samples[i]["segment"] == "segment_1"
        for key in expected_sample:
            np.testing.assert_equal(ts_samples[i][key], expected_sample[key])
            if "categorical" in key:
                for column in ts_samples[i][key]:
                    assert ts_samples[i][key][column].base is not None
            else:
                assert ts_samples[i][key].base is not None


def test_mlp_forward_fail_nans():
    batch = {
        "decoder_real": torch.Tensor([[[torch.nan, 2, 3], [1, 2, 3], [1, 2, 3]]]),
        "decoder_target": torch.Tensor([[[1], [2], [3]]]),
        "segment": "A",
    }
    model = MLPNet(input_size=3, hidden_size=[1], embedding_sizes={}, lr=1e-2, loss=nn.MSELoss(), optimizer_params=None)
    with pytest.raises(ValueError, match="There are NaNs in features"):
        _ = model.forward(batch)


def test_mlp_step():

    batch = {
        "decoder_real": torch.Tensor([[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]),
        "decoder_categorical": {},
        "decoder_target": torch.Tensor([[[1], [2], [3]]]),
        "segment": "A",
    }
    model = MLPNet(input_size=3, hidden_size=[1], embedding_sizes={}, lr=1e-2, loss=nn.MSELoss(), optimizer_params=None)
    loss, decoder_target, output = model.step(batch)
    assert type(loss) == torch.Tensor
    assert type(decoder_target) == torch.Tensor
    assert torch.all(decoder_target == batch["decoder_target"])
    assert type(output) == torch.Tensor
    assert output.shape == torch.Size([1, 3, 1])


def test_mlp_step_fail_nans():
    batch = {
        "decoder_real": torch.Tensor([[torch.nan, 2, 3], [1, 2, 3], [1, 2, 3]]),
        "decoder_target": torch.Tensor([[1], [2], [3]]),
        "segment": "A",
    }
    model = MLPNet(input_size=3, hidden_size=[1], embedding_sizes={}, lr=1e-2, loss=nn.MSELoss(), optimizer_params=None)
    with pytest.raises(ValueError, match="There are NaNs in features"):
        _ = model.step(batch)


def test_mlp_layers():
    model = MLPNet(input_size=3, hidden_size=[10], embedding_sizes={}, lr=1e-2, loss=None, optimizer_params=None)
    model_ = nn.Sequential(
        nn.Linear(in_features=3, out_features=10), nn.ReLU(), nn.Linear(in_features=10, out_features=1)
    )
    assert repr(model_) == repr(model.mlp)


def test_save_load(example_tsds):
    horizon = 3
    model = MLPModel(
        input_size=9,
        hidden_size=[10],
        lr=1e-1,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
    )
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    transforms = [lag, fourier, std]
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=transforms, horizon=horizon)


@pytest.mark.parametrize(
    "model",
    [
        MLPModel(
            input_size=9,
            hidden_size=[5],
            lr=1e-1,
            decoder_length=14,
            trainer_params=dict(max_epochs=1),
        ),
        MLPModel(
            input_size=9,
            hidden_size=[5, 5],
            lr=1e-1,
            decoder_length=14,
            trainer_params=dict(max_epochs=1),
        ),
        MLPModel(
            input_size=9,
            hidden_size=[5, 5, 5],
            lr=1e-1,
            decoder_length=14,
            trainer_params=dict(max_epochs=1),
        ),
    ],
)
def test_params_to_tune(model, example_tsds):
    ts = example_tsds
    horizon = 3
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    transforms = [lag, fourier, std]
    ts.fit_transform(transforms)
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)


@patch("torch.backends.mps.is_available", return_value=True)
def test_mlp_warning_training_with_mps(mock_is_available):
    with pytest.warns(
        UserWarning,
        match="If you use MPS sometimes it can cause unexpected results.\
                               If this happens try setting `accelerate=cpu` in trainer_params.",
    ):
        decoder_length = 14
        model = MLPModel(
            input_size=10,
            hidden_size=[10, 10, 10, 10, 10],
            lr=1e-1,
            decoder_length=decoder_length,
            trainer_params=dict(max_epochs=100),
        )
