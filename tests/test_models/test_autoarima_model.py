from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from etna.models import AutoARIMAModel
from etna.pipeline import Pipeline
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_prediction_components_are_present


def _check_forecast(ts, model, horizon):
    model.fit(ts)
    future_ts = ts.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res["target"].isnull().values.any()
    assert len(res) == horizon * 2


def _check_predict(ts, model):
    model.fit(ts)
    res = model.predict(ts)
    res = res.to_pandas(flatten=True)

    assert not res["target"].isnull().values.any()
    assert len(res) == len(ts.timestamps) * 2


def test_fit_with_exogs_warning(ts_with_non_regressor_exog):
    ts = ts_with_non_regressor_exog
    model = AutoARIMAModel()
    with pytest.warns(UserWarning, match="This model doesn't work with exogenous features unknown in future"):
        model.fit(ts)


def test_fit_str_category_fail(ts_with_non_convertable_category_regressor):
    model = AutoARIMAModel()
    ts = ts_with_non_convertable_category_regressor
    with pytest.raises(ValueError, match="Only convertible to float features are allowed"):
        model.fit(ts)


def test_save_regressors_on_fit(example_reg_tsds):
    model = AutoARIMAModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_select_regressors_correctly_datetime_timestamp(example_reg_tsds):
    ts = example_reg_tsds
    model = AutoARIMAModel()
    model.fit(ts=ts)
    for segment, segment_model in model._models.items():
        segment_features = ts[:, segment, :].droplevel("segment", axis=1).reset_index()

        segment_regressors_expected = segment_features[ts.regressors].astype(float)
        segment_regressors_expected.index = segment_features["timestamp"]
        segment_regressors = segment_model._select_regressors(df=segment_features)

        pd.testing.assert_frame_equal(segment_regressors, segment_regressors_expected)


def test_select_regressors_correctly_int_timestamp(example_reg_tsds_int_timestamp):
    ts = example_reg_tsds_int_timestamp
    model = AutoARIMAModel()
    model.fit(ts=ts)
    for segment, segment_model in model._models.items():
        segment_features = ts[:, segment, :].droplevel("segment", axis=1).reset_index()

        segment_regressors_expected = segment_features[ts.regressors].astype(float)
        segment_regressors_expected.index = pd.Index(np.arange(len(segment_regressors_expected)), name="timestamp")
        segment_regressors = segment_model._select_regressors(df=segment_features)

        pd.testing.assert_frame_equal(segment_regressors, segment_regressors_expected)


@pytest.mark.parametrize("ts_name", ["example_tsds", "example_tsds_int_timestamp"])
def test_prediction(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    _check_forecast(ts=deepcopy(ts), model=AutoARIMAModel(), horizon=7)
    _check_predict(ts=deepcopy(ts), model=AutoARIMAModel())


@pytest.mark.parametrize("ts_name", ["example_reg_tsds", "example_reg_tsds_int_timestamp"])
def test_prediction_with_reg(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    _check_forecast(ts=deepcopy(ts), model=AutoARIMAModel(), horizon=7)
    _check_predict(ts=deepcopy(ts), model=AutoARIMAModel())


@pytest.mark.filterwarnings("ignore: Error fitting  ARIMA")
def test_forecast_with_short_regressors_fail(ts_with_short_regressor):
    ts = ts_with_short_regressor
    with pytest.raises(ValueError, match="Regressors .* contain NaN values"):
        _check_forecast(ts=deepcopy(ts), model=AutoARIMAModel(), horizon=20)


def test_prediction_with_params(example_reg_tsds):
    horizon = 7
    model = AutoARIMAModel(
        start_p=3,
        start_q=3,
        max_p=4,
        max_d=4,
        max_q=5,
        start_P=2,
        start_Q=2,
        max_P=3,
        max_D=3,
        max_Q=2,
        max_order=6,
        m=2,
        seasonal=True,
    )
    _check_forecast(ts=deepcopy(example_reg_tsds), model=deepcopy(model), horizon=horizon)
    _check_predict(ts=deepcopy(example_reg_tsds), model=deepcopy(model))


@pytest.mark.parametrize("ts_name", ["example_tsds", "example_tsds_int_timestamp"])
@pytest.mark.parametrize("method_name", ["forecast", "predict"])
def test_prediction_interval_insample(ts_name, method_name, request):
    ts = request.getfixturevalue(ts_name)
    model = AutoARIMAModel()
    model.fit(ts)
    method = getattr(model, method_name)
    forecast = method(ts, prediction_interval=True, quantiles=[0.025, 0.975])

    assert forecast.prediction_intervals_names == ("target_0.025", "target_0.975")
    prediction_intervals = forecast.get_prediction_intervals()
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        # N.B. inplace forecast will not change target values, because `combine_first` in `SARIMAXModel.forecast` only fill nan values
        # assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        # assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()

        segment_intervals = prediction_intervals[segment]
        assert np.allclose(segment_slice["target_0.975"], segment_intervals["target_0.975"])
        assert np.allclose(segment_slice["target_0.025"], segment_intervals["target_0.025"])


@pytest.mark.parametrize("ts_name", ["example_tsds", "example_tsds_int_timestamp"])
def test_forecast_prediction_interval_infuture(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    model = AutoARIMAModel()
    model.fit(ts)
    future = ts.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])

    assert forecast.prediction_intervals_names == ("target_0.025", "target_0.975")
    prediction_intervals = forecast.get_prediction_intervals()

    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()

        segment_intervals = prediction_intervals[segment]
        assert np.allclose(segment_slice["target_0.975"], segment_intervals["target_0.975"])
        assert np.allclose(segment_slice["target_0.025"], segment_intervals["target_0.025"])


@pytest.mark.parametrize("method_name", ["forecast", "predict"])
def test_prediction_raise_error_if_not_fitted(example_tsds, method_name):
    """Test that AutoARIMA raise error when calling prediction without being fit."""
    model = AutoARIMAModel()
    with pytest.raises(ValueError, match="model is not fitted!"):
        method = getattr(model, method_name)
        _ = method(ts=example_tsds)


def test_get_model_before_training():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = AutoARIMAModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_after_training(example_tsds):
    """Check that get_model method returns dict of objects of AutoARIMA class."""
    pipeline = Pipeline(model=AutoARIMAModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], SARIMAXResultsWrapper)


def test_forecast_1_point(example_tsds):
    """Check that AutoARIMA work with 1 point forecast."""
    horizon = 1
    model = AutoARIMAModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    pred = model.forecast(future_ts)
    assert len(pred.df) == horizon
    pred_quantiles = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.8])
    assert len(pred_quantiles.df) == horizon


def test_save_load(example_tsds):
    model = AutoARIMAModel()
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_prediction_decomposition(outliers_tsds):
    train, test = outliers_tsds.train_test_split(test_size=10)
    assert_prediction_components_are_present(model=AutoARIMAModel(), train=train, test=test)


def test_params_to_tune():
    model = AutoARIMAModel()
    assert len(model.params_to_tune()) == 0
