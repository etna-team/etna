import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_anomalies_prediction_interval
from etna.analysis.outliers.prediction_interval_outliers import create_ts_by_column
from etna.datasets import TSDataset
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from tests.utils import convert_ts_to_int_timestamp


@pytest.fixture
def outliers_tsds_int_timestamp(outliers_tsds) -> TSDataset:
    return convert_ts_to_int_timestamp(ts=outliers_tsds, shift=10)


@pytest.fixture
def outliers_tsds_with_external_timestamp(outliers_tsds) -> TSDataset:
    df = outliers_tsds.to_pandas(flatten=True)
    df_exog = df.copy()
    df_exog["external_timestamp"] = df["timestamp"]
    df_exog.drop(columns=["target"], inplace=True)
    df_wide = TSDataset.to_dataset(df.drop(columns=["exog"])).iloc[1:-1]
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=pd.offsets.Day(), known_future=["external_timestamp"])
    ts = convert_ts_to_int_timestamp(ts=ts, shift=10)
    return ts


@pytest.mark.parametrize("column", ["exog"])
def test_create_ts_by_column_interface(outliers_tsds, column):
    """Test that `create_ts_column` produces correct columns."""
    new_ts = create_ts_by_column(outliers_tsds, column)
    assert isinstance(new_ts, TSDataset)
    assert outliers_tsds.segments == new_ts.segments
    assert new_ts.features == ["target"]


@pytest.mark.parametrize("column", ["exog"])
def test_create_ts_by_column_retain_column(outliers_tsds, column):
    """Test that `create_ts_column` selects correct data in selected columns."""
    new_ts = create_ts_by_column(outliers_tsds, column)
    for segment in new_ts.segments:
        new_series = new_ts[:, segment, "target"]
        original_series = outliers_tsds[:, segment, column]
        new_series = new_series[~new_series.isna()]
        original_series = original_series[~original_series.isna()]
        assert np.all(new_series == original_series)


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize("model", (ProphetModel, SARIMAXModel))
def test_get_anomalies_prediction_interval_interface(outliers_tsds, model, in_column):
    """Test that `get_anomalies_prediction_interval` produces correct columns."""
    anomalies = get_anomalies_prediction_interval(outliers_tsds, model=model, interval_width=0.95, in_column=in_column)
    assert isinstance(anomalies, dict)
    assert sorted(anomalies.keys()) == sorted(outliers_tsds.segments)
    for segment in anomalies.keys():
        assert isinstance(anomalies[segment], list)
        for date in anomalies[segment]:
            assert isinstance(date, np.datetime64)


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize(
    "model, interval_width, ts_name, true_anomalies",
    (
        (
            ProphetModel,
            0.95,
            "outliers_tsds",
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
        (
            SARIMAXModel,
            0.999,
            "outliers_tsds",
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
        (
            SARIMAXModel,
            0.999,
            "outliers_tsds_int_timestamp",
            {"1": [20], "2": [18, 36]},
        ),
    ),
)
def test_get_anomalies_prediction_interval_values(model, interval_width, ts_name, true_anomalies, in_column, request):
    """Test that `get_anomalies_prediction_interval` generates correct values."""
    ts = request.getfixturevalue(ts_name)
    predicted_anomalies = get_anomalies_prediction_interval(
        ts, model=model, interval_width=interval_width, in_column=in_column
    )
    assert predicted_anomalies == true_anomalies


def test_get_anomalies_prediction_interval_values_prophet_with_external_timestamp(
    outliers_tsds_with_external_timestamp,
):
    ts = outliers_tsds_with_external_timestamp
    predicted_anomalies = get_anomalies_prediction_interval(
        ts, model=ProphetModel, interval_width=0.95, in_column="target", timestamp_column="external_timestamp"
    )
    true_anomalies = {"1": [19], "2": [17, 35]}
    assert predicted_anomalies == true_anomalies


@pytest.mark.parametrize(
    "model, interval_width, in_column",
    (
        (ProphetModel, 0.95, "target"),
        (SARIMAXModel, 0.999, "target"),
    ),
)
def test_get_anomalies_prediction_interval_imbalanced_tsdf(imbalanced_tsdf, model, interval_width, in_column):
    get_anomalies_prediction_interval(imbalanced_tsdf, model=model, interval_width=interval_width, in_column=in_column)


@pytest.mark.parametrize("index_only, value_type", ((True, list), (False, pd.Series)))
def test_get_anomalies_prediction_interval_index_only(outliers_tsds, index_only, value_type):
    result = get_anomalies_prediction_interval(
        outliers_tsds, model=ProphetModel, interval_width=0.95, in_column="target", index_only=index_only
    )

    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, value_type)
