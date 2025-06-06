from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from prophet.serialize import model_to_dict

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.models import ProphetModel
from etna.models.prophet import _ProphetAdapter
from etna.pipeline import Pipeline
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_prediction_components_are_present
from tests.test_models.utils import assert_sampling_is_valid
from tests.test_models.utils import check_forecast_context_ignorant
from tests.test_models.utils import check_predict_context_ignorant


def test_fit_str_category_fail(ts_with_non_convertable_category_regressor):
    model = ProphetModel()
    ts = ts_with_non_convertable_category_regressor
    with pytest.raises(ValueError, match="Only convertible to numeric features are allowed"):
        model.fit(ts)


def test_fit_with_exogs_warning(ts_with_non_regressor_exog):
    ts = ts_with_non_regressor_exog
    model = ProphetModel()
    with pytest.warns(UserWarning, match="This model doesn't work with exogenous features unknown in future"):
        model.fit(ts)


def test_fit_int_timestamp_fail(example_tsds_int_timestamp):
    ts = example_tsds_int_timestamp
    model = ProphetModel()
    with pytest.raises(ValueError, match="Invalid timestamp! Only datetime type is supported."):
        model.fit(ts)


def test_fit_external_timestamp_not_present_fail(example_tsds):
    ts = example_tsds
    model = ProphetModel(timestamp_column="unknown_feature")
    with pytest.raises(ValueError, match="Invalid timestamp_column! It isn't present in a given dataset."):
        model.fit(ts)


def test_fit_external_timestamp_not_regressor_fail():
    df = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_wide = TSDataset.to_dataset(df)
    df_exog = generate_ar_df(periods=100, start_time=10, n_segments=1, freq=None)
    df_exog.rename(columns={"target": "external_timestamp"}, inplace=True)
    df_exog["external_timestamp"] = pd.date_range(start="2020-01-01", periods=100)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, known_future=[], freq=None)

    model = ProphetModel(timestamp_column="external_timestamp")
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

    model = ProphetModel(timestamp_column="external_timestamp")
    with pytest.raises(ValueError, match="Invalid timestamp_column! Only datetime type is supported."):
        model.fit(ts)


@pytest.mark.parametrize(
    "ts_name, timestamp_column",
    [
        ("example_tsds", None),
        ("example_reg_tsds", None),
        ("ts_with_external_timestamp", "external_timestamp"),
    ],
)
def test_prediction(ts_name, timestamp_column, request):
    ts = request.getfixturevalue(ts_name)
    check_forecast_context_ignorant(ts=deepcopy(ts), model=ProphetModel(timestamp_column=timestamp_column), horizon=7)
    check_predict_context_ignorant(ts=deepcopy(ts), model=ProphetModel(timestamp_column=timestamp_column))


def test_prediction_with_cap_floor():
    cap = 101
    floor = -1

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=100),
            "segment": "segment_0",
            "target": list(range(100)),
        }
    )
    df_exog = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=120),
            "segment": "segment_0",
            "cap": cap,
            "floor": floor,
        }
    )
    ts = TSDataset(
        df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog), freq=pd.offsets.Day(), known_future="all"
    )

    model = ProphetModel(growth="logistic")
    pipeline = Pipeline(model=model, horizon=7)
    pipeline.fit(ts)

    ts_future = pipeline.forecast()
    df_future = ts_future.to_pandas(flatten=True)

    assert np.all(df_future["target"] < cap)


def test_forecast_with_short_regressors_fail(ts_with_short_regressor):
    ts = ts_with_short_regressor
    with pytest.raises(ValueError, match="Regressors .* contain NaN values"):
        check_forecast_context_ignorant(ts=deepcopy(ts), model=ProphetModel(), horizon=20)


@pytest.mark.parametrize(
    "ts_name, timestamp_column",
    [
        ("example_tsds", None),
        ("ts_with_external_timestamp", "external_timestamp"),
    ],
)
def test_prediction_interval_run_insample(ts_name, timestamp_column, request):
    ts = request.getfixturevalue(ts_name)
    model = ProphetModel(timestamp_column=timestamp_column)
    model.fit(ts)
    forecast = model.forecast(ts, prediction_interval=True, quantiles=[0.025, 0.975])

    assert forecast.prediction_intervals_names == ("target_0.025", "target_0.975")
    prediction_intervals = forecast.get_prediction_intervals()
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()

        segment_intervals = prediction_intervals[segment]
        assert np.allclose(segment_slice["target_0.975"], segment_intervals["target_0.975"])
        assert np.allclose(segment_slice["target_0.025"], segment_intervals["target_0.025"])


@pytest.mark.parametrize(
    "ts_name, timestamp_column",
    [
        ("example_tsds", None),
        ("ts_with_external_timestamp", "external_timestamp"),
    ],
)
def test_prediction_interval_run_infuture(ts_name, timestamp_column, request):
    ts = request.getfixturevalue(ts_name)
    model = ProphetModel(timestamp_column=timestamp_column)
    model.fit(ts)
    future = ts.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])

    assert forecast.prediction_intervals_names == ("target_0.025", "target_0.975")
    prediction_intervals = forecast.get_prediction_intervals()
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()

        segment_intervals = prediction_intervals[segment]
        assert np.allclose(segment_slice["target_0.975"], segment_intervals["target_0.975"])
        assert np.allclose(segment_slice["target_0.025"], segment_intervals["target_0.025"])


def test_save_regressors_on_fit(example_reg_tsds):
    model = ProphetModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_get_model_before_training():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = ProphetModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_after_training(example_tsds):
    """Check that get_model method returns dict of objects of Prophet class."""
    pipeline = Pipeline(model=ProphetModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], Prophet)


@pytest.fixture
def prophet_default_params():
    params = {
        "growth": "linear",
        "changepoints": None,
        "n_changepoints": 25,
        "changepoint_range": 0.8,
        "yearly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto",
        "holidays": None,
        "seasonality_mode": "additive",
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.05,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000,
        "stan_backend": None,
        "additional_seasonality_params": (),
    }
    return params


def test_getstate_not_fitted(prophet_default_params):
    model = _ProphetAdapter()
    state = model.__getstate__()
    expected_state = {
        "_is_fitted": False,
        "_model_dict": {},
        "regressor_columns": None,
        "timestamp_column": None,
        **prophet_default_params,
    }
    assert state == expected_state


def test_getstate_fitted(example_tsds, prophet_default_params):
    model = _ProphetAdapter()
    df = example_tsds.to_pandas()["segment_1"].reset_index()
    model.fit(df, regressors=[])
    state = model.__getstate__()
    expected_state = {
        "_is_fitted": True,
        "_model_dict": model_to_dict(model.model),
        "regressor_columns": [],
        "timestamp_column": None,
        **prophet_default_params,
    }
    assert state == expected_state


def test_setstate_not_fitted():
    model_1 = _ProphetAdapter(n_changepoints=25)
    initial_state = model_1.__getstate__()

    model_2 = _ProphetAdapter(n_changepoints=20)
    model_2.__setstate__(initial_state)
    new_state = model_2.__getstate__()
    assert new_state == initial_state


def test_setstate_fitted(example_tsds):
    model_1 = _ProphetAdapter()
    df = example_tsds.to_pandas()["segment_1"].reset_index()
    model_1.fit(df, regressors=[])
    initial_state = model_1.__getstate__()

    model_2 = _ProphetAdapter()
    model_2.__setstate__(initial_state)
    new_state = model_2.__getstate__()
    assert new_state == initial_state


def test_save_load(example_tsds):
    model = ProphetModel()
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.mark.parametrize(
    "custom_seasonality",
    (
        [{"name": "s1", "period": 14, "fourier_order": 3}],
        [{"name": "s1", "period": 14, "fourier_order": 3}, {"name": "s2", "period": 10, "fourier_order": 3}],
    ),
)
def test_custom_seasonality(custom_seasonality):
    model = ProphetModel(additional_seasonality_params=custom_seasonality)
    for seasonality in custom_seasonality:
        assert seasonality["name"] in model._base_model.model.seasonalities


@pytest.fixture
def prophet_dfs(dfs_w_exog):
    df = pd.concat(dfs_w_exog, axis=0)
    df["cap"] = 4.0

    h1_mask = np.arange(len(df)) % 3 == 0
    h2_mask = np.arange(len(df)) % 5 == 0

    h1 = pd.DataFrame(
        {
            "holiday": "h1",
            "ds": df["timestamp"][h1_mask],
            "lower_window": 0,
            "upper_window": 1,
        }
    )

    h2 = pd.DataFrame(
        {
            "holiday": "h2",
            "ds": df["timestamp"][h2_mask],
            "lower_window": 0,
            "upper_window": 1,
        }
    )
    holidays = pd.concat([h1, h2]).reset_index(drop=True)

    return df.iloc[-60:-20], df.iloc[-20:], holidays


def test_check_mul_components_not_fitted_error():
    model = _ProphetAdapter()
    with pytest.raises(ValueError, match="This model is not fitted!"):
        model._check_mul_components()


def test_prepare_prophet_df_regressors_not_set_error(prophet_dfs):
    _, test, _ = prophet_dfs
    model = _ProphetAdapter()
    with pytest.raises(ValueError, match="List of regressor is not set!"):
        model._prepare_prophet_df(df=test)


@pytest.mark.parametrize(
    "seasonality_mode,custom_seasonality",
    (
        ("multiplicative", [{"name": "s1", "period": 14, "fourier_order": 1, "mode": "additive"}]),
        ("multiplicative", []),
        ("additive", [{"name": "s1", "period": 14, "fourier_order": 1, "mode": "multiplicative"}]),
    ),
)
def test_check_mul_components(prophet_dfs, seasonality_mode, custom_seasonality):
    _, test, _ = prophet_dfs

    model = _ProphetAdapter(seasonality_mode=seasonality_mode, additional_seasonality_params=custom_seasonality)
    model.fit(df=test, regressors=["f1", "f2"])

    with pytest.raises(ValueError, match="Forecast decomposition is only supported for additive components!"):
        model.predict_components(df=test)


@pytest.mark.parametrize(
    "regressors,regressors_comps", ((["f1", "f2", "cap"], ["target_component_f1", "target_component_f2"]), ([], []))
)
@pytest.mark.parametrize(
    "custom_seas,custom_seas_comp",
    (
        ([{"name": "s1", "period": 14, "fourier_order": 1}], ["target_component_s1"]),
        ([], []),
    ),
)
@pytest.mark.parametrize("use_holidays,holidays_comp", ((True, ["target_component_holidays"]), (False, [])))
@pytest.mark.parametrize("daily,daily_comp", ((True, ["target_component_daily"]), (False, [])))
@pytest.mark.parametrize("weekly,weekly_comp", ((True, ["target_component_weekly"]), (False, [])))
@pytest.mark.parametrize("yearly,yearly_comp", ((True, ["target_component_yearly"]), (False, [])))
def test_predict_components_names(
    prophet_dfs,
    regressors,
    regressors_comps,
    use_holidays,
    holidays_comp,
    daily,
    daily_comp,
    weekly,
    weekly_comp,
    yearly,
    yearly_comp,
    custom_seas,
    custom_seas_comp,
):
    _, test, holidays = prophet_dfs

    if not use_holidays:
        holidays = None

    expected_columns = set(
        regressors_comps
        + holidays_comp
        + daily_comp
        + weekly_comp
        + yearly_comp
        + custom_seas_comp
        + ["target_component_trend"]
    )

    model = _ProphetAdapter(
        holidays=holidays,
        daily_seasonality=daily,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        additional_seasonality_params=custom_seas,
    )
    model.fit(df=test, regressors=regressors)

    components = model.predict_components(df=test)

    assert set(components.columns) == expected_columns


@pytest.mark.parametrize(
    "timestamp_column,timestamp_column_regressors", [(None, []), ("external_timestamp", ["external_timestamp"])]
)
@pytest.mark.parametrize("growth,cap", (("linear", []), ("logistic", ["cap"])))
@pytest.mark.parametrize("regressors", (["f1", "f2"], []))
@pytest.mark.parametrize("custom_seas", ([{"name": "s1", "period": 14, "fourier_order": 1}], []))
@pytest.mark.parametrize("use_holidays", (True, False))
@pytest.mark.parametrize("daily", (True, False))
@pytest.mark.parametrize("weekly", (True, False))
@pytest.mark.parametrize("yearly", (True, False))
def test_predict_components_sum_up_to_target(
    prophet_dfs,
    regressors,
    use_holidays,
    daily,
    weekly,
    yearly,
    custom_seas,
    growth,
    cap,
    timestamp_column,
    timestamp_column_regressors,
):
    train, test, holidays = prophet_dfs

    if not use_holidays:
        holidays = None

    if timestamp_column is not None:
        train[timestamp_column] = train["timestamp"]
        test[timestamp_column] = test["timestamp"]

    model = _ProphetAdapter(
        growth=growth,
        holidays=holidays,
        daily_seasonality=daily,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        additional_seasonality_params=custom_seas,
        timestamp_column=timestamp_column,
    )
    model.fit(df=train, regressors=regressors + cap + timestamp_column_regressors)

    components = model.predict_components(df=test)
    pred = model.predict(df=test)

    np.testing.assert_allclose(np.sum(components, axis=1), pred["target"].values)


def test_prediction_decomposition(outliers_tsds):
    train, test = outliers_tsds.train_test_split(test_size=10)
    assert_prediction_components_are_present(model=ProphetModel(), train=train, test=test)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = ProphetModel()
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
