from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearPerSegmentModel
from etna.transforms.timestamp import FourierTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


def add_seasonality(series: pd.Series, period: int, magnitude: float) -> pd.Series:
    """Add seasonality to given series."""
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += np.sin(2 * np.pi * indices / period) * magnitude
    return new_series


def get_one_df(period_1, period_2, magnitude_1, magnitude_2):
    timestamp = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")
    df = pd.DataFrame({"timestamp": timestamp})
    target = 0
    indices = np.arange(timestamp.shape[0])
    target += np.sin(2 * np.pi * indices * 2 / period_1) * magnitude_1
    target += np.cos(2 * np.pi * indices * 3 / period_2) * magnitude_2
    target += np.random.normal(scale=0.05, size=timestamp.shape[0])
    df["target"] = target
    return df


@pytest.fixture
def example_ts(example_df):
    df = TSDataset.to_dataset(example_df)
    example_df["external_timestamp"] = example_df["timestamp"]
    example_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(example_df)
    ts = TSDataset(df=df, df_exog=df_exog, freq="H")
    return ts


@pytest.fixture
def example_ts_int_timestamp(example_df):
    example_df["timestamp"] = (example_df["timestamp"] - example_df["timestamp"].min()) // pd.Timedelta("1H")
    df = TSDataset.to_dataset(example_df)
    example_df["external_timestamp"] = example_df["timestamp"]
    example_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(example_df)
    ts = TSDataset(df=df, df_exog=df_exog, freq=None)
    return ts


@pytest.fixture
def example_ts_with_regressor(example_ts):
    df = example_ts.raw_df
    df_exog = example_ts.df_exog
    ts = TSDataset(df=df.iloc[:-10], df_exog=df_exog, freq=example_ts.freq, known_future=["external_timestamp"])
    return ts


@pytest.fixture
def example_ts_with_nans(example_ts_int_timestamp) -> TSDataset:
    ts = example_ts_int_timestamp
    df = ts.raw_df
    df_exog = ts.df_exog
    df_exog.loc[df_exog.index[:3], pd.IndexSlice[:, "external_timestamp"]] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


@pytest.fixture
def ts_trend_seasonal(random_seed) -> TSDataset:
    df_1 = get_one_df(period_1=7, period_2=30.4, magnitude_1=1, magnitude_2=1 / 2)
    df_1["segment"] = "segment_1"
    df_2 = get_one_df(period_1=7, period_2=30.4, magnitude_1=1 / 2, magnitude_2=1 / 5)
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq="D")


@pytest.mark.parametrize("order, mods", [(None, [1, 2, 3, 4]), (2, None)])
def test_repr(order, mods):
    transform = FourierTransform(
        period=10,
        order=order,
        mods=mods,
    )
    transform_repr = transform.__repr__()
    true_repr = f"FourierTransform(period = 10, order = {order}, mods = {mods}, out_column = None, in_column = None, )"
    assert transform_repr == true_repr


@pytest.mark.parametrize("period", [-1, 0, 1, 1.5])
def test_fail_period(period):
    """Test that transform is not created with wrong period."""
    with pytest.raises(ValueError, match="Period should be at least 2"):
        _ = FourierTransform(period=period, order=1)


@pytest.mark.parametrize("order", [0, 5])
def test_fail_order(order):
    """Test that transform is not created with wrong order."""
    with pytest.raises(ValueError, match="Order should be within"):
        _ = FourierTransform(period=7, order=order)


@pytest.mark.parametrize("mods", [[0], [0, 1, 2, 3], [1, 2, 3, 7], [7]])
def test_fail_mods(mods):
    """Test that transform is not created with wrong mods."""
    with pytest.raises(ValueError, match="Every mod should be within"):
        _ = FourierTransform(period=7, mods=mods)


def test_fail_set_none():
    """Test that transform is not created without order and mods."""
    with pytest.raises(ValueError, match="There should be exactly one option set"):
        _ = FourierTransform(period=7)


def test_fail_set_both():
    """Test that transform is not created with both order and mods set."""
    with pytest.raises(ValueError, match="There should be exactly one option set"):
        _ = FourierTransform(period=7, order=1, mods=[1, 2, 3])


@pytest.mark.parametrize(
    "period, order, num_columns", [(6, 2, 4), (7, 2, 4), (6, 3, 5), (7, 3, 6), (5.5, 2, 4), (5.5, 3, 5)]
)
def test_column_names(example_ts, period, order, num_columns):
    """Test that transform creates expected number of columns and they can be recreated by its name."""
    segments = example_ts.columns.get_level_values("segment").unique()
    initial_columns = example_ts.columns.get_level_values("feature").unique()
    transform = FourierTransform(period=period, order=order)

    transformed_df = transform.fit_transform(deepcopy(example_ts)).to_pandas()
    new_columns = transformed_df.columns.get_level_values("feature").unique().difference(initial_columns)

    assert len(new_columns) == num_columns
    for column in new_columns:
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(deepcopy(example_ts)).to_pandas()
        columns_temp = df_temp.columns.get_level_values("feature").unique().difference(initial_columns)

        assert len(columns_temp) == 1
        generated_column = columns_temp[0]
        assert generated_column == column
        assert np.all(
            df_temp.loc[:, pd.IndexSlice[segments, generated_column]]
            == transformed_df.loc[:, pd.IndexSlice[segments, column]]
        )


def test_column_names_out_column(example_ts):
    """Test that transform creates expected columns if `out_column` is set"""
    initial_columns = example_ts.columns.get_level_values("feature").unique()
    transform = FourierTransform(period=10, order=3, out_column="regressor_fourier")
    transformed_df = transform.fit_transform(example_ts).to_pandas()
    columns = transformed_df.columns.get_level_values("feature").unique().difference(initial_columns)
    expected_columns = {f"regressor_fourier_{i}" for i in range(1, 7)}
    assert set(columns) == expected_columns


def test_transform_in_column_datetime_fail(example_ts):
    transform = FourierTransform(period=10, order=3, out_column="regressor_fourier", in_column="external_timestamp")
    transform.fit(example_ts)
    with pytest.raises(ValueError, match="Only numeric data is supported if in_column is set"):
        _ = transform.transform(example_ts)


def test_transform_datetime_index_fail_not_fitted(example_ts):
    transform = FourierTransform(period=10, order=3, out_column="regressor_fourier")
    with pytest.raises(ValueError, match="The transform isn't fitted"):
        _ = transform.transform(example_ts)


@pytest.mark.parametrize(
    "in_column, ts_name",
    [
        (None, "example_ts"),
        (None, "example_ts_int_timestamp"),
        ("external_timestamp", "example_ts_int_timestamp"),
    ],
)
@pytest.mark.parametrize("period, mod", [(24, 1), (24, 2), (24, 9), (24, 20), (24, 23), (7.5, 3), (7.5, 4)])
def test_transform_values(in_column, ts_name, period, mod, request):
    """Test that transform generates correct values."""
    ts = request.getfixturevalue(ts_name)
    transform = FourierTransform(period=period, mods=[mod], out_column="regressor_fourier", in_column=in_column)
    transformed_df = transform.fit_transform(ts).to_pandas()

    for segment in ts.segments:
        transform_values = transformed_df.loc[:, pd.IndexSlice[segment, f"regressor_fourier_{mod}"]]
        elapsed = np.arange(len(ts.index)) / period
        order = (mod + 1) // 2
        if mod % 2 == 0:
            expected_values = np.cos(2 * np.pi * order * elapsed)
        else:
            expected_values = np.sin(2 * np.pi * order * elapsed)

        assert np.allclose(transform_values, expected_values, atol=1e-12)


@pytest.mark.parametrize(
    "shift",
    [
        0,
        3,
    ],
)
@pytest.mark.parametrize(
    "in_column, ts_name",
    [
        (None, "example_ts"),
        (None, "example_ts_int_timestamp"),
        ("external_timestamp", "example_ts_int_timestamp"),
    ],
)
@pytest.mark.parametrize("period, mod", [(24, 1), (24, 2), (24, 9), (24, 20), (24, 23), (7.5, 3), (7.5, 4)])
def test_transform_values_with_shift(shift, in_column, ts_name, period, mod, request):
    ts = request.getfixturevalue(ts_name)

    ts_1 = ts
    ts_2 = TSDataset(df=ts.raw_df.iloc[shift:], df_exog=ts.df_exog, freq=ts.freq, known_future=ts.known_future)
    transform = FourierTransform(period=period, mods=[mod], out_column="regressor_fourier", in_column=in_column)

    transform.fit(ts)
    transformed_df_1 = transform.transform(ts_1).to_pandas()
    transformed_df_2 = transform.transform(ts_2).to_pandas()

    for segment in ts.segments:
        transform_values_1 = transformed_df_1.loc[:, pd.IndexSlice[segment, f"regressor_fourier_{mod}"]].iloc[shift:]
        transform_values_2 = transformed_df_2.loc[:, pd.IndexSlice[segment, f"regressor_fourier_{mod}"]]
        pd.testing.assert_series_equal(transform_values_1, transform_values_2)


def test_transform_values_with_nans(example_ts_with_nans):
    initial_columns = example_ts_with_nans.columns.get_level_values("feature").unique()
    transform = FourierTransform(period=10, order=3, out_column="regressor_fourier", in_column="external_timestamp")
    transformed_df = transform.fit_transform(example_ts_with_nans).to_pandas()

    new_columns = transformed_df.columns.get_level_values("feature").unique().difference(initial_columns)
    for segment in example_ts_with_nans.segments:
        result_df = transformed_df.loc[:, pd.IndexSlice[segment, new_columns]]
        assert np.all(result_df.isna().sum() == 3)


def test_get_regressors_info_index(example_ts):
    transform = FourierTransform(period=10, order=3, out_column="fourier")

    regressors_info = transform.get_regressors_info()

    expected_regressor_info = [f"{transform.out_column}_{mod}" for mod in range(1, 7)]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


def test_get_regressors_info_in_column_fail_not_fitted(example_ts):
    transform = FourierTransform(period=10, order=3, out_column="fourier", in_column="external_timestamp")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


def test_get_regressors_info_in_column_fitted_exog(example_ts):
    transform = FourierTransform(period=10, order=3, out_column="fourier", in_column="external_timestamp")

    transform.fit(example_ts)
    regressors_info = transform.get_regressors_info()

    assert regressors_info == []


def test_get_regressors_info_in_column_fitted_regressor(example_ts_with_regressor):
    transform = FourierTransform(period=10, order=3, out_column="fourier", in_column="external_timestamp")

    transform.fit(example_ts_with_regressor)
    regressors_info = transform.get_regressors_info()

    expected_regressor_info = [f"{transform.out_column}_{mod}" for mod in range(1, 7)]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


def test_forecast(ts_trend_seasonal):
    """Test that transform works correctly in forecast."""
    transform_1 = FourierTransform(period=7, order=3)
    transform_2 = FourierTransform(period=30.4, order=5)
    transforms = [transform_1, transform_2]
    ts_train, ts_test = ts_trend_seasonal.train_test_split(test_size=10)
    ts_train.fit_transform(transforms=transforms)
    model = LinearPerSegmentModel()
    model.fit(ts_train)
    ts_future = ts_train.make_future(10, transforms=transforms)
    ts_forecast = model.forecast(ts_future)
    ts_forecast.inverse_transform(transforms)
    metric = R2("macro")
    r2 = metric(ts_test, ts_forecast)
    assert r2 > 0.95


def test_save_load(ts_trend_seasonal):
    transform = FourierTransform(period=7, order=3)
    assert_transformation_equals_loaded_original(transform=transform, ts=ts_trend_seasonal)


@pytest.mark.parametrize(
    "transform, expected_length",
    [
        (FourierTransform(period=7, order=1), 1),
        (FourierTransform(period=7, mods=[1]), 0),
        (FourierTransform(period=7, mods=[1, 4]), 0),
        (FourierTransform(period=30.4, order=1), 1),
        (FourierTransform(period=365.25, order=1), 1),
    ],
)
def test_params_to_tune(transform, expected_length, ts_trend_seasonal):
    ts = ts_trend_seasonal
    assert len(transform.params_to_tune()) == expected_length
    assert_sampling_is_valid(transform=transform, ts=ts)
