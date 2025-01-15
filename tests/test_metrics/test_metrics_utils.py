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
        "MAPE(mode = 'macro', missing_mode = 'error', eps = 1e-05, )",
    ]
    result = compute_metrics(metrics=metrics, y_true=true_df, y_pred=forecast_df)
    np.testing.assert_array_equal(sorted(expected_keys), sorted(result.keys()))


@pytest.fixture
def metrics_df_with_folds() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "segment": ["segment_0"] * 3 + ["segment_1"] * 3 + ["segment_2"] * 3,
            "MAE": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            "MSE": [None, 3.0, 4.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            "MAPE": [None, None, None, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0],
            "SMAPE": [None, None, None, None, None, None, 50.0, 60.0, 70.0],
            "RMSE": [None, None, None, None, None, None, None, None, None],
            "fold_number": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    return df


@pytest.fixture
def metrics_df_no_folds(metrics_df_with_folds) -> pd.DataFrame:
    df = metrics_df_with_folds
    df = df.groupby("segment").mean(numeric_only=False).reset_index().drop("fold_number", axis=1)
    return df


@pytest.fixture
def aggregated_metrics_df() -> Dict[str, Any]:
    result = {
        "MAE_mean": 3.0,
        "MAE_median": 3.0,
        "MAE_std": 0.816496580927726,
        "MAE_notna_size": 3.0,
        "MAE_percentile_5": 2.1,
        "MAE_percentile_25": 2.5,
        "MAE_percentile_75": 3.5,
        "MAE_percentile_95": 3.9,
        "MSE_mean": 4.5,
        "MSE_median": 4.0,
        "MSE_std": 1.0801234497346435,
        "MSE_notna_size": 3.0,
        "MSE_percentile_5": 3.55,
        "MSE_percentile_25": 3.75,
        "MSE_percentile_75": 5.0,
        "MSE_percentile_95": 5.8,
        "MAPE_mean": 35.0,
        "MAPE_median": 35.0,
        "MAPE_std": 5.0,
        "MAPE_notna_size": 2.0,
        "MAPE_percentile_5": 30.5,
        "MAPE_percentile_25": 32.5,
        "MAPE_percentile_75": 37.5,
        "MAPE_percentile_95": 39.5,
        "SMAPE_mean": 60.0,
        "SMAPE_median": 60.0,
        "SMAPE_std": 0.0,
        "SMAPE_notna_size": 1.0,
        "SMAPE_percentile_5": 60.0,
        "SMAPE_percentile_25": 60.0,
        "SMAPE_percentile_75": 60.0,
        "SMAPE_percentile_95": 60.0,
        "RMSE_mean": None,
        "RMSE_median": None,
        "RMSE_std": None,
        "RMSE_notna_size": 0.0,
        "RMSE_percentile_5": None,
        "RMSE_percentile_25": None,
        "RMSE_percentile_75": None,
        "RMSE_percentile_95": None,
    }
    return result


@pytest.mark.parametrize(
    "df_name, answer_name",
    [
        ("metrics_df_with_folds", "aggregated_metrics_df"),
        ("metrics_df_no_folds", "aggregated_metrics_df"),
    ],
)
def test_aggregate_metrics_df(df_name, answer_name, request):
    metrics_df = request.getfixturevalue(df_name)
    answer = request.getfixturevalue(answer_name)
    result = aggregate_metrics_df(metrics_df)
    assert result == answer
