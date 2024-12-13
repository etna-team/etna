from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.metrics import MAPE
from etna.metrics import MSE
from etna.metrics.utils import aggregate_metrics_df
from etna.metrics.utils import compute_metrics


def test_compute_metrics(train_test_dfs: Tuple[TSDataset, TSDataset]):
    """Check that compute_metrics return correct metrics keys."""
    forecast_df, true_df = train_test_dfs
    metrics = [MAE("per-segment"), MAE(mode="macro"), MSE("per-segment"), MAPE(mode="macro", eps=1e-5)]
    expected_keys = [
        "MAE(mode = 'per-segment', missing_mode = 'error', )",
        "MAE(mode = 'macro', missing_mode = 'error', )",
        "MSE(mode = 'per-segment', missing_mode = 'error', )",
        "MAPE(mode = 'macro', eps = 1e-05, )",
    ]
    result = compute_metrics(metrics=metrics, y_true=true_df, y_pred=forecast_df)
    np.testing.assert_array_equal(sorted(expected_keys), sorted(result.keys()))


@pytest.fixture
def metrics_df_with_folds() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "segment": ["segment_0"] * 3 + ["segment_1"] * 3 + ["segment_2"] * 3,
            "MAE": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            "MSE": [2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            "fold_number": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    return df


@pytest.fixture
def metrics_df_no_folds(metrics_df_with_folds) -> pd.DataFrame:
    df = metrics_df_with_folds
    df = df.groupby("segment").mean().reset_index().drop("fold_number", axis=1)
    return df


@pytest.fixture
def aggregated_metrics_df() -> Dict[str, Any]:
    result = {
        "MAE_median": 3.0,
        "MAE_mean": 3.0,
        "MAE_std": 0.816496580927726,
        "MAE_size": 3.0,
        "MAE_percentile_5": 2.1,
        "MAE_percentile_25": 2.5,
        "MAE_percentile_75": 3.5,
        "MAE_percentile_95": 3.9,
        "MSE_median": 4.0,
        "MSE_mean": 4.333333333333333,
        "MSE_std": 1.247219128924647,
        "MSE_size": 3.0,
        "MSE_percentile_5": 3.1,
        "MSE_percentile_25": 3.5,
        "MSE_percentile_75": 5.0,
        "MSE_percentile_95": 5.8,
    }
    return result


@pytest.fixture
def metrics_df_with_folds_with_missing() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "segment": ["segment_0"] * 3 + ["segment_1"] * 3 + ["segment_2"] * 3,
            "MAE": [None, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            "MSE": [2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            "fold_number": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    return df


@pytest.fixture
def metrics_df_no_folds_with_missing(metrics_df_with_folds_with_missing) -> pd.DataFrame:
    df = metrics_df_with_folds_with_missing
    df = (
        df.groupby("segment")
        .apply(lambda x: x.mean(skipna=False, numeric_only=False))
        .reset_index()
        .drop("fold_number", axis=1)
    )
    return df


@pytest.fixture
def aggregated_metrics_df_with_missing() -> Dict[str, Any]:
    result = {
        "MAE_mean": 3.5,
        "MAE_median": 3.5,
        "MAE_std": 0.5,
        "MAE_size": 2.0,
        "MAE_percentile_5": 3.05,
        "MAE_percentile_25": 3.25,
        "MAE_percentile_75": 3.75,
        "MAE_percentile_95": 3.95,
        "MSE_mean": 4.333333333333333,
        "MSE_median": 4.0,
        "MSE_std": 1.247219128924647,
        "MSE_size": 3.0,
        "MSE_percentile_5": 3.1,
        "MSE_percentile_25": 3.5,
        "MSE_percentile_75": 5.0,
        "MSE_percentile_95": 5.8,
    }
    return result


@pytest.fixture
def metrics_df_with_folds_with_full_missing() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "segment": ["segment_0"] * 3 + ["segment_1"] * 3 + ["segment_2"] * 3,
            "MAE": [None, 2.0, 3.0, 2.0, None, 4.0, 3.0, 4.0, None],
            "MSE": [2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            "fold_number": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    return df


@pytest.fixture
def metrics_df_no_folds_with_full_missing(metrics_df_with_folds_with_full_missing) -> pd.DataFrame:
    df = metrics_df_with_folds_with_full_missing
    df = (
        df.groupby("segment")
        .apply(lambda x: x.mean(skipna=False, numeric_only=False))
        .reset_index()
        .drop("fold_number", axis=1)
    )
    return df


@pytest.fixture
def aggregated_metrics_df_with_full_missing() -> Dict[str, Any]:
    result = {
        "MAE_mean": None,
        "MAE_median": None,
        "MAE_std": None,
        "MAE_size": 0.0,
        "MAE_percentile_5": None,
        "MAE_percentile_25": None,
        "MAE_percentile_75": None,
        "MAE_percentile_95": None,
        "MSE_mean": 4.333333333333333,
        "MSE_median": 4.0,
        "MSE_std": 1.247219128924647,
        "MSE_size": 3.0,
        "MSE_percentile_5": 3.1,
        "MSE_percentile_25": 3.5,
        "MSE_percentile_75": 5.0,
        "MSE_percentile_95": 5.8,
    }
    return result


@pytest.mark.parametrize(
    "df_name, answer_name",
    [
        ("metrics_df_with_folds", "aggregated_metrics_df"),
        ("metrics_df_no_folds", "aggregated_metrics_df"),
        ("metrics_df_with_folds_with_missing", "aggregated_metrics_df_with_missing"),
        ("metrics_df_no_folds_with_missing", "aggregated_metrics_df_with_missing"),
        ("metrics_df_with_folds_with_full_missing", "aggregated_metrics_df_with_full_missing"),
        ("metrics_df_no_folds_with_full_missing", "aggregated_metrics_df_with_full_missing"),
    ],
)
def test_aggregate_metrics_df(df_name, answer_name, request):
    metrics_df = request.getfixturevalue(df_name)
    answer = request.getfixturevalue(answer_name)
    result = aggregate_metrics_df(metrics_df)
    assert result == answer
