import numpy as np
import pandas as pd
import pytest

from etna.metrics import MAE
from etna.models import BATSModel
from etna.models import CatBoostPerSegmentModel
from etna.models import DeadlineMovingAverageModel
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models import SimpleExpSmoothingModel
from etna.models import TBATSModel
from etna.pipeline import Pipeline
from etna.transforms import IForestOutlierTransform
from etna.transforms import LagTransform
from etna.transforms import ModelDecomposeTransform
from etna.transforms import TimeSeriesImputerTransform


def simple_pipeline_with_decompose(in_column, horizon):
    pipeline = Pipeline(
        transforms=[ModelDecomposeTransform(model=HoltWintersModel(), in_column=in_column)],
        model=HoltWintersModel(),
        horizon=horizon,
    )
    return pipeline


@pytest.mark.parametrize("in_column", ("target", "feat"))
def test_init(in_column):
    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column=in_column)
    assert transform.required_features == [in_column]
    assert transform._first_timestamp is None
    assert transform._last_timestamp is None


def test_unsupported_model():
    with pytest.raises(ValueError, match=".* is not supported! Supported models are:"):
        ModelDecomposeTransform(model=CatBoostPerSegmentModel())


def test_prepare_ts_invalid_feature(simple_tsdf):
    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column="feat")
    with pytest.raises(KeyError, match="is not found in features"):
        _ = transform._prepare_ts(ts=simple_tsdf)


def test_is_not_fitted(simple_tsdf):
    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column="feat")
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        transform.transform(ts=simple_tsdf)


def test_prepare_ts_in_column_target(ts_with_exogs):
    ts = ts_with_exogs

    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column="target")
    prepared_ts = transform._prepare_ts(ts=ts)

    assert prepared_ts is not ts
    assert prepared_ts._df_exog is None
    pd.testing.assert_frame_equal(prepared_ts._df, ts[..., "target"])


@pytest.mark.parametrize(
    "ts_name,in_column",
    (
        ("outliers_df_with_two_columns", "feature"),
        ("ts_with_exogs", "exog"),
        ("ts_with_exogs", "holiday"),
    ),
)
def test_prepare_ts_in_column_feature(ts_name, in_column, request):
    ts = request.getfixturevalue(ts_name)

    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column=in_column)
    prepared_ts = transform._prepare_ts(ts=ts)

    assert prepared_ts is not ts
    assert "feature" not in prepared_ts.features
    assert prepared_ts._df_exog is None
    pd.testing.assert_frame_equal(
        prepared_ts._df, ts[..., in_column].rename({in_column: "target"}, axis=1, level="feature")
    )


@pytest.mark.parametrize(
    "ts_name,in_column",
    (
        ("outliers_df_with_two_columns", "target"),
        ("outliers_df_with_two_columns", "feature"),
        ("ts_with_exogs", "target"),
        ("ts_with_exogs", "exog"),
        ("ts_with_exogs", "holiday"),
        ("example_tsds_int_timestamp", "target"),
    ),
)
def test_fit(ts_name, in_column, request):
    ts = request.getfixturevalue(ts_name)
    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column=in_column)
    transform.fit(ts=ts)

    assert transform._first_timestamp == ts.timestamps.min()
    assert transform._last_timestamp == ts.timestamps.max()


@pytest.mark.parametrize("residuals", (True, False))
@pytest.mark.parametrize("in_column", ("target", "exog"))
def test_add_residuals(ts_with_exogs, residuals, in_column):
    ts = ts_with_exogs

    transform = ModelDecomposeTransform(model=HoltWintersModel(), in_column=in_column, residuals=residuals)
    transformed = transform.fit_transform(ts=ts)

    assert (f"{in_column}_residuals" in transformed.features) is residuals


def test_timestamp_from_future(ts_with_exogs_train_test):
    train, test = ts_with_exogs_train_test
    transform = ModelDecomposeTransform(model=HoltWintersModel())
    transform.fit_transform(train)

    with pytest.raises(ValueError, match="Dataset to be transformed must contain historical observations in range"):
        transform.transform(test)


def test_timestamp_from_history(ts_with_exogs_train_test):
    test, train = ts_with_exogs_train_test
    transform = ModelDecomposeTransform(model=HoltWintersModel())
    transform.fit_transform(train)

    with pytest.raises(ValueError, match="First index of the dataset to be transformed must be larger"):
        transform.transform(test)


@pytest.mark.parametrize(
    "in_column",
    (
        "target",
        "holiday",
        "exog",
    ),
)
@pytest.mark.parametrize("horizon", (1, 5))
def test_simple_pipeline_forecast(ts_with_exogs, in_column, horizon):
    ts = ts_with_exogs

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon)

    pipeline.fit(ts=ts)
    forecast = pipeline.forecast()

    assert forecast.size()[0] == horizon
    assert np.sum(forecast[..., "target"].isna().sum()) == 0


@pytest.mark.parametrize(
    "in_column",
    (
        "target",
        "holiday",
        "exog",
    ),
)
@pytest.mark.parametrize("horizon", (1, 5))
def test_simple_pipeline_predict(ts_with_exogs, in_column, horizon):
    ts = ts_with_exogs

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon)

    pipeline.fit(ts=ts)
    forecast = pipeline.predict(ts)

    assert forecast.size()[0] == ts.size()[0]
    assert np.sum(forecast[..., "target"].isna().sum()) == 0


@pytest.mark.parametrize(
    "in_column",
    (
        "target",
        "holiday",
        "exog",
    ),
)
@pytest.mark.parametrize("horizon", (1, 5))
def test_simple_pipeline_predict_components(ts_with_exogs, in_column, horizon):
    ts = ts_with_exogs

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon)

    pipeline.fit(ts=ts)
    forecast = pipeline.predict(ts, return_components=True)

    assert forecast.size()[0] == ts.size()[0]
    assert forecast.target_components_names == ("target_component_level",)


@pytest.mark.parametrize(
    "in_column",
    (
        "target",
        "holiday",
        "exog",
    ),
)
@pytest.mark.parametrize("horizon", (1, 5))
def test_simple_pipeline_backtest(ts_with_exogs, in_column, horizon):
    ts = ts_with_exogs

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon)

    forecast_ts_list = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=3)["forecasts"]
    forecast_df = pd.concat([forecast.to_pandas() for forecast in forecast_ts_list], axis=0)

    assert len(forecast_df) == horizon * 3
    assert np.sum(forecast_df.loc[:, pd.IndexSlice[:, "target"]].isna().sum()) == 0


@pytest.mark.parametrize(
    "ts_name,in_column",
    (
        ("outliers_df_with_two_columns", "target"),
        ("outliers_df_with_two_columns", "feature"),
        ("ts_with_exogs", "target"),
        ("ts_with_exogs", "exog"),
    ),
)
@pytest.mark.parametrize(
    "decompose_model",
    (
        HoltWintersModel(),
        ProphetModel(),
    ),
)
@pytest.mark.parametrize(
    "forecast_model, forecast_transforms",
    [
        (HoltWintersModel(), []),
        (ProphetModel(), []),
        (CatBoostPerSegmentModel(iterations=10), [LagTransform(in_column="target", lags=[3, 4, 5])]),
    ],
)
def test_pipeline_models(ts_name, in_column, decompose_model, forecast_model, forecast_transforms, request):
    ts = request.getfixturevalue(ts_name)

    pipeline = Pipeline(
        transforms=forecast_transforms + [ModelDecomposeTransform(model=decompose_model, in_column=in_column)],
        model=forecast_model,
        horizon=3,
    )

    pipeline.fit(ts)
    forecast = pipeline.forecast()

    assert forecast.size()[0] == 3
    assert np.sum(forecast._df.loc[:, pd.IndexSlice[:, "target"]].isna().sum()) == 0


@pytest.mark.parametrize(
    "decompose_model",
    (
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
        HoltWintersModel(trend="add", seasonal="add"),
        ProphetModel(),
        SARIMAXModel(),
        DeadlineMovingAverageModel(),
        SeasonalMovingAverageModel(),
        BATSModel(use_arma_errors=False),
        TBATSModel(use_arma_errors=False),
    ),
)
def test_decompose_models(ts_with_exogs, decompose_model):
    pipeline = Pipeline(
        transforms=[ModelDecomposeTransform(model=decompose_model, in_column="exog")],
        model=CatBoostPerSegmentModel(iterations=10),
        horizon=3,
    )

    pipeline.fit(ts_with_exogs)
    forecast = pipeline.forecast()

    assert forecast.size()[0] == 3
    assert np.sum(forecast._df.loc[:, pd.IndexSlice[:, "target"]].isna().sum()) == 0


@pytest.mark.parametrize("answer", ({"1": ["2021-01-11"], "2": ["2021-01-27"]},))
def test_outlier_detection(outliers_solid_tsds, answer):
    ts = outliers_solid_tsds

    transforms = [
        ModelDecomposeTransform(
            model=HoltWintersModel(seasonal="add", seasonal_periods=3), in_column="target", residuals=True
        ),
        IForestOutlierTransform(
            in_column="target",
            features_to_use=["target_residuals", "target_seasonality", "target_level"],
            contamination=0.01,
        ),
    ]
    ts.fit_transform(transforms)

    for segment in ts.segments:
        empty_values = pd.isna(ts[:, segment, "target"])
        assert empty_values.sum() == len(answer[segment])
        assert all(empty_values[answer[segment]])


def test_outlier_detection_pipeline(outliers_solid_tsds):
    ts = outliers_solid_tsds
    pipeline = Pipeline(
        transforms=[
            ModelDecomposeTransform(model=HoltWintersModel(), in_column="target"),
            IForestOutlierTransform(in_column="target"),
            TimeSeriesImputerTransform(in_column="target"),
        ],
        model=SARIMAXModel(),
        horizon=3,
    )
    pipeline.fit(ts)


@pytest.mark.parametrize(
    "decompose_model, context_size",
    (
        (HoltWintersModel(), 0),
        (ProphetModel(), 0),
        (SARIMAXModel(), 0),
        (SeasonalMovingAverageModel(window=3, seasonality=1), 3),
        (BATSModel(use_arma_errors=False, use_trend=True), 0),
        (TBATSModel(use_arma_errors=False, use_trend=True), 0),
    ),
)
def test_stride_transform(forward_stride_datasets, decompose_model, context_size):
    train, test = forward_stride_datasets

    transform = ModelDecomposeTransform(model=decompose_model, residuals=True)

    transform.fit(train)
    transformed = transform.transform(test)

    assert not transformed._df.iloc[context_size:10].isna().any().any()
    assert transformed._df.iloc[10:].isna().all().any()


@pytest.mark.parametrize("features_to_ignore", (["target"], ["regressor_1"], ["target", "regressor_1"]))
def test_repetitive_call_before_pipeline_with_ignore(outliers_solid_tsds, features_to_ignore):
    """Problem arises when decomposition transform is applied the second time to the same `TSDataset`.
    This test mimics behavior when we do analysis before building pipeline and want to apply results in pipeline
    transforms without copying the dataset.

    In such cases error `KeyError: '[] not found in axis'` may be raised by pandas.
    """
    ts = outliers_solid_tsds

    ts.fit_transform(
        [
            ModelDecomposeTransform(
                model=HoltWintersModel(trend="add", seasonal="add", seasonal_periods=None), residuals=True
            )
        ]
    )

    pipeline = Pipeline(
        transforms=[
            ModelDecomposeTransform(
                model=HoltWintersModel(trend="add", seasonal="add", seasonal_periods=None), residuals=True
            ),
            IForestOutlierTransform(
                in_column="target",
                features_to_ignore=features_to_ignore,
            ),
            TimeSeriesImputerTransform(),
        ],
        model=SARIMAXModel(),
    )

    pipeline.fit(ts)
