import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import CatBoostPerSegmentModel
from etna.models import HoltWintersModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import FourierDecomposeTransform
from etna.transforms import IForestOutlierTransform
from etna.transforms import LagTransform
from etna.transforms import TimeSeriesImputerTransform


def simple_pipeline_with_decompose(in_column, horizon, k):
    pipeline = Pipeline(
        transforms=[FourierDecomposeTransform(k=k, in_column=in_column)],
        model=HoltWintersModel(),
        horizon=horizon,
    )
    return pipeline


@pytest.fixture()
def ts_with_missing(ts_with_exogs):
    target_df = ts_with_exogs[..., "target"]
    target_df.iloc[10] = np.nan

    return TSDataset(df=target_df, freq=ts_with_exogs.freq)


@pytest.mark.parametrize("in_column", ("target", "feat"))
def test_init(in_column):
    transform = FourierDecomposeTransform(k=5, in_column=in_column)
    assert transform.required_features == [in_column]
    assert transform._first_timestamp is None
    assert transform._last_timestamp is None


@pytest.mark.parametrize("k", (-1, 0))
def test_invalid_k(k):
    with pytest.raises(ValueError, match="Parameter `k` must be positive integer!"):
        FourierDecomposeTransform(k=k, in_column="target")


@pytest.mark.parametrize(
    "series, answ",
    (
        (pd.Series([1]), 1),
        (pd.Series([1, 2]), 2),
        (pd.Series([1, 2, 3]), 2),
        (pd.Series([1, 2, 3, 4]), 3),
        (pd.Series([1, 2, 3, 4, 5]), 3),
        (pd.Series([1, 2, 3, 4, 5, 6]), 4),
    ),
)
def test_get_num_pos_freqs(series, answ):
    res = FourierDecomposeTransform._get_num_pos_freqs(series=series)
    assert res == answ


def test_check_segments_missing_values(ts_with_missing):
    df = ts_with_missing[..., "target"]
    transform = FourierDecomposeTransform(k=5)
    with pytest.raises(ValueError, match=f"Feature `target` contains missing values"):
        transform._check_segments(df=df)


@pytest.mark.parametrize("k", (52, 100))
def test_check_segments_large_k(ts_with_exogs, k):
    df = ts_with_exogs[..., "target"]
    transform = FourierDecomposeTransform(k=k)
    with pytest.raises(ValueError, match=f"Parameter `k` must not be greater then"):
        transform._check_segments(df=df)


def test_check_segments_ok(ts_with_exogs):
    df = ts_with_exogs[..., "target"]
    transform = FourierDecomposeTransform(k=5)
    transform._check_segments(df=df)


@pytest.mark.parametrize(
    "series",
    (
        pd.Series(np.arange(5)),
        pd.Series(np.arange(10)),
        pd.Series([np.nan] * 2 + list(range(5)) + [np.nan] * 3),
    ),
)
def test_fft_components_out_format(series):
    expected_columns = ["dft_0", "dft_1", "dft_2", "dft_residuals"]
    transform = FourierDecomposeTransform(k=3, residuals=True)

    decompose_df = transform._dft_components(series=series)

    assert isinstance(decompose_df, pd.DataFrame)
    pd.testing.assert_index_equal(decompose_df.index, series.index)
    assert (decompose_df.columns == expected_columns).all()
    np.testing.assert_allclose(np.sum(decompose_df.values, axis=1), series.values)


def test_is_not_fitted(simple_tsdf):
    transform = FourierDecomposeTransform(k=5, in_column="feat")
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        transform.transform(ts=simple_tsdf)


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
    transform = FourierDecomposeTransform(k=5, in_column=in_column)
    transform.fit(ts=ts)

    assert transform._first_timestamp == ts.timestamps.min()
    assert transform._last_timestamp == ts.timestamps.max()


@pytest.mark.parametrize("residuals", (True, False))
@pytest.mark.parametrize("in_column", ("target", "exog"))
def test_add_residuals(ts_with_exogs, residuals, in_column):
    ts = ts_with_exogs

    transform = FourierDecomposeTransform(k=5, in_column=in_column, residuals=residuals)
    transformed = transform.fit_transform(ts=ts)

    assert (f"{in_column}_dft_residuals" in transformed.features) is residuals


def test_timestamp_from_history(ts_with_exogs_train_test):
    test, train = ts_with_exogs_train_test
    transform = FourierDecomposeTransform(k=5)
    transform.fit_transform(train)

    with pytest.raises(ValueError, match="First index of the dataset to be transformed must be larger"):
        transform.transform(test)


def test_timestamp_from_future(ts_with_exogs_train_test):
    train, test = ts_with_exogs_train_test
    transform = FourierDecomposeTransform(k=5)
    transform.fit_transform(train)

    with pytest.raises(ValueError, match="Dataset to be transformed must contain historical observations in range"):
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

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon, k=5)

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

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon, k=5)

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

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon, k=5)

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

    pipeline = simple_pipeline_with_decompose(in_column=in_column, horizon=horizon, k=5)

    forecast_ts_list = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=3)["forecasts"]
    forecast_df = pd.concat([forecast.to_pandas() for forecast in forecast_ts_list])

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
@pytest.mark.parametrize("k", (1, 5, 10, 40, 51))
@pytest.mark.parametrize(
    "forecast_model, forecast_transforms",
    [
        (ProphetModel(), []),
        (CatBoostPerSegmentModel(iterations=10), [LagTransform(in_column="target", lags=[3, 4, 5])]),
    ],
)
def test_pipeline_parameter_k(ts_name, in_column, forecast_model, forecast_transforms, k, request):
    ts = request.getfixturevalue(ts_name)

    pipeline = Pipeline(
        transforms=forecast_transforms + [FourierDecomposeTransform(k=5, in_column=in_column)],
        model=forecast_model,
        horizon=3,
    )

    pipeline.fit(ts)
    forecast = pipeline.forecast()

    assert forecast.size()[0] == 3
    assert np.sum(forecast._df.loc[:, pd.IndexSlice[:, "target"]].isna().sum()) == 0


@pytest.mark.parametrize("answer", ({"1": ["2021-01-11"], "2": ["2021-01-09"]},))
def test_outlier_detection(outliers_solid_tsds, answer):
    ts = outliers_solid_tsds

    transforms = [
        FourierDecomposeTransform(k=2, in_column="target", residuals=True),
        IForestOutlierTransform(
            in_column="target",
            features_to_ignore=["target", "regressor_1"],
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
            FourierDecomposeTransform(k=5, in_column="target"),
            IForestOutlierTransform(in_column="target"),
            TimeSeriesImputerTransform(in_column="target"),
        ],
        model=ProphetModel(),
        horizon=3,
    )
    pipeline.fit(ts)


@pytest.mark.parametrize("k", (1, 5))
def test_stride_transform(forward_stride_datasets, k):
    train, test = forward_stride_datasets

    transform = FourierDecomposeTransform(k=k, residuals=True)

    transform.fit(train)
    transformed = transform.transform(test)

    assert not transformed._df.iloc[:10].isna().any().any()
    assert transformed._df.iloc[10:].isna().all().any()


def test_repetitive_fit_transform(ts_with_exogs):
    transform = FourierDecomposeTransform(in_column="target", k=3)

    ts_with_exogs = transform.fit_transform(ts_with_exogs)
    columns_before = ts_with_exogs._df.columns.tolist()

    ts_with_exogs = transform.inverse_transform(ts_with_exogs)

    transform = FourierDecomposeTransform(in_column="target", k=3)

    ts_with_exogs = transform.fit_transform(ts_with_exogs)
    columns_after = ts_with_exogs._df.columns.tolist()
    assert columns_before == columns_after
