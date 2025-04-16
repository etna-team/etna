import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_residuals
from etna.analysis.forecast.utils import _get_existing_intervals
from etna.analysis.forecast.utils import _select_prediction_intervals_names
from etna.analysis.forecast.utils import _validate_intersecting_folds
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def residuals():
    timestamp = pd.date_range("2020-01-01", periods=100, freq=pd.offsets.Day())
    df = pd.DataFrame(
        {
            "timestamp": timestamp.tolist() * 2,
            "segment": ["segment_0"] * len(timestamp) + ["segment_1"] * len(timestamp),
            "target": np.arange(len(timestamp)).tolist() + (np.arange(len(timestamp)) + 1).tolist(),
        }
    )
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq=pd.offsets.Day())

    forecast_df = df[df["timestamp"] >= timestamp[10]]
    forecast_df.loc[df["segment"] == "segment_0", "target"] = -1
    forecast_df.loc[df["segment"] == "segment_1", "target"] = 1
    forecast_ts_list = [
        TSDataset(df=forecast_df.loc[df["timestamp"] < timestamp[55]], freq=pd.offsets.Day()),
        TSDataset(df=forecast_df.loc[df["timestamp"] >= timestamp[55]], freq=pd.offsets.Day()),
    ]

    residuals_df = ts[ts.timestamps[10:], :, :]
    residuals_df.loc[:, pd.IndexSlice["segment_0", "target"]] += 1
    residuals_df.loc[:, pd.IndexSlice["segment_1", "target"]] -= 1

    return residuals_df, forecast_ts_list, ts


@pytest.fixture
def residuals_with_components(residuals):
    residuals_df, forecast_df, ts = residuals
    df_wide = ts.to_pandas()
    df_component_1 = df_wide.rename(columns={"target": "component_1"}, level="feature")
    df_component_2 = df_wide.rename(columns={"target": "component_2"}, level="feature")
    df_component_1.loc[:, pd.IndexSlice[:, "component_1"]] *= 0.7
    df_component_2.loc[:, pd.IndexSlice[:, "component_2"]] *= 0.3
    df_components = pd.concat([df_component_1, df_component_2], axis=1)
    ts.add_target_components(df_components)
    return residuals_df, forecast_df, ts


@pytest.fixture
def residuals_with_intervals(residuals):
    residuals_df, forecast_df, ts = residuals
    intervals = generate_ar_df(periods=100, n_segments=2, start_time="2020-01-01", freq=pd.offsets.Day())
    intervals.rename(columns={"target": "target_0.025"}, inplace=True)
    intervals_wide = TSDataset.to_dataset(intervals)
    ts.add_prediction_intervals(intervals_wide)
    return residuals_df, forecast_df, ts


@pytest.fixture
def dataset_dict(toy_dataset_equal_targets_and_quantiles):
    return {"1": toy_dataset_equal_targets_and_quantiles}


def test_get_residuals(residuals):
    """Test that get_residuals finds residuals correctly."""
    residuals_df, forecast_ts_list, ts = residuals
    actual_residuals = get_residuals(forecast_ts_list=forecast_ts_list, ts=ts)
    assert actual_residuals.to_pandas().equals(residuals_df)


def test_get_residuals_with_components(residuals_with_components):
    """Test that get_residuals finds residuals correctly in case of target components presence."""
    residuals_df, forecast_ts_list, ts = residuals_with_components
    actual_residuals = get_residuals(forecast_ts_list=forecast_ts_list, ts=ts)
    assert actual_residuals.to_pandas().equals(residuals_df)


def test_get_residuals_with_invervals(residuals_with_intervals):
    """Test that get_residuals deletes prediction intervals."""
    residuals_df, forecast_ts_list, ts = residuals_with_intervals
    actual_residuals = get_residuals(forecast_ts_list=forecast_ts_list, ts=ts)
    assert "target_0.025" not in actual_residuals.features
    assert "target_0.025" not in actual_residuals.prediction_intervals_names
    pd.testing.assert_frame_equal(actual_residuals._df, residuals_df)


def test_get_residuals_not_matching_lengths(residuals):
    """Test that get_residuals fails to find residuals correctly if ts hasn't answers."""
    residuals_df, forecast_ts_list, ts = residuals
    ts = TSDataset(df=ts[ts.timestamps[:-10], :, :], freq=pd.offsets.Day())
    with pytest.raises(KeyError):
        _ = get_residuals(forecast_ts_list=forecast_ts_list, ts=ts)


def test_get_residuals_not_matching_segments(residuals):
    """Test that get_residuals fails to find residuals correctly if segments of dataset and forecast differ."""
    residuals_df, forecast_ts_list, ts = residuals
    for forecast_ts in forecast_ts_list:
        columns_frame = forecast_ts._df.columns.to_frame()
        columns_frame["segment"] = ["segment_0", "segment_3"]
        forecast_ts._df.columns = pd.MultiIndex.from_frame(columns_frame)
    with pytest.raises(KeyError, match="Segments of `ts` and `forecast_df` should be the same"):
        _ = get_residuals(forecast_ts_list=forecast_ts_list, ts=ts)


@pytest.mark.parametrize(
    "fold_numbers",
    [
        pd.Series([0, 0, 1, 1, 2, 2], index=pd.date_range("2020-01-01", periods=6, freq=pd.offsets.Day())),
        pd.Series([0, 0, 1, 1, 2, 2], index=pd.date_range("2020-01-01", periods=6, freq=pd.offsets.Day(2))),
        pd.Series([2, 2, 0, 0, 1, 1], index=pd.date_range("2020-01-01", periods=6, freq=pd.offsets.Day())),
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-06"),
            ],
        ),
    ],
)
def test_validate_intersecting_segments_ok(fold_numbers):
    _validate_intersecting_folds(fold_numbers)


@pytest.mark.parametrize(
    "fold_numbers",
    [
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
        ),
        pd.Series([0, 0, 1, 1, 0, 0], index=pd.date_range("2020-01-01", periods=6, freq=pd.offsets.Day())),
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
            ],
        ),
        pd.Series(
            [1, 1, 0, 0],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
            ],
        ),
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-08"),
            ],
        ),
        pd.Series(
            [1, 1, 0, 0],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-08"),
            ],
        ),
    ],
)
def test_validate_intersecting_segments_fail(fold_numbers):
    with pytest.raises(ValueError):
        _validate_intersecting_folds(fold_numbers)


@pytest.mark.parametrize("ts_name", ("example_tsds", "toy_dataset_equal_targets_and_quantiles"))
def test_get_existing_intervals(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    assert _get_existing_intervals(ts) == set(ts.prediction_intervals_names)


@pytest.mark.parametrize("quantiles", (None, [0.01]))
def test_select_prediction_intervals_names(dataset_dict, quantiles):
    selected_borders = _select_prediction_intervals_names(forecast_results=dataset_dict, quantiles=quantiles)
    assert selected_borders == ["target_0.01"]


@pytest.mark.parametrize("quantiles", ([0.001], [0.1, 0.9]))
def test_select_prediction_intervals_names_non_existing_quantiles_error(dataset_dict, quantiles):
    with pytest.raises(ValueError, match="Unable to find provided quantiles"):
        _ = _select_prediction_intervals_names(forecast_results=dataset_dict, quantiles=quantiles)


@pytest.mark.parametrize("quantiles", ([0.001, 0.01], [0.01, 0.1, 0.9]))
def test_select_prediction_intervals_names_extra_quantiles(dataset_dict, quantiles):
    with pytest.warns(UserWarning, match="Quantiles .* do not exist in each forecast dataset."):
        _ = _select_prediction_intervals_names(forecast_results=dataset_dict, quantiles=quantiles)
