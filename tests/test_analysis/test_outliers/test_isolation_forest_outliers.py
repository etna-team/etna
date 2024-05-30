import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from etna.analysis.outliers.isolation_forest_outliers import _get_anomalies_isolation_forest_segment
from etna.analysis.outliers.isolation_forest_outliers import _prepare_segment_df
from etna.analysis.outliers.isolation_forest_outliers import _select_features
from etna.analysis.outliers.isolation_forest_outliers import get_anomalies_isolation_forest
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def ts_with_features():
    df = generate_ar_df(n_segments=2, periods=5, start_time="2000-01-01")
    df["target"] = [np.NAN, np.NAN, 1, 2, 3] + [np.NAN, 10, np.NAN, 30, 40]
    df["exog_1"] = [1] * 5 + [2] * 5
    ts = TSDataset(df=df.drop(columns=["exog_1"]), freq="D", df_exog=df.drop(columns=["target"]))
    return ts


@pytest.fixture
def df_segment_0():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-04"), pd.Timestamp("2000-01-05")],
            "target": [1.0, 2.0, 3.0],
            "exog_1": [1, 1, 1],
        }
    )
    return df


@pytest.fixture
def df_segment_1():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-04"), pd.Timestamp("2000-01-05")],
            "target": [10.0, 30.0, 40.0],
            "exog_1": [2, 2, 2],
        }
    )
    return df


@pytest.mark.parametrize(
    "features_to_use,features_to_ignore,expected_features",
    [
        (None, None, ["target", "exog_1"]),
        (["exog_1"], None, ["exog_1"]),
        (None, ["exog_1"], ["target"]),
    ],
)
def test_select_features(ts_with_features, features_to_use, features_to_ignore, expected_features):
    df = _select_features(ts=ts_with_features, features_to_use=features_to_use, features_to_ignore=features_to_ignore)
    features = set(df.columns.get_level_values("feature"))
    assert sorted(features) == sorted(expected_features)


@pytest.mark.parametrize(
    "features_to_use,features_to_ignore,expected_error",
    [
        (["exog_1"], ["exog_1"], "There should be exactly one option set: features_to_use or features_to_ignore"),
        (["exog_2"], None, "Features {'exog_2'} are not present in the dataset."),
        (None, ["exog_2"], "Features {'exog_2'} are not present in the dataset."),
    ],
)
def test_select_features_fails(ts_with_features, features_to_use, features_to_ignore, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        _ = _select_features(
            ts=ts_with_features, features_to_use=features_to_use, features_to_ignore=features_to_ignore
        )


@pytest.mark.parametrize(
    "segment,ignore_missing, expected_df",
    [
        ("segment_0", True, "df_segment_0"),
        ("segment_1", True, "df_segment_1"),
        ("segment_0", False, "df_segment_0"),
    ],
)
def test_prepare_segment_df(ts_with_features, segment, ignore_missing, expected_df, request):
    expected_df = request.getfixturevalue(expected_df)
    df = _prepare_segment_df(df=ts_with_features.to_pandas(), segment=segment, ignore_missing=ignore_missing)
    pd.testing.assert_frame_equal(df.sort_index(axis=1), expected_df.sort_index(axis=1))


def test_prepare_segment_df_fails(ts_with_features):
    with pytest.raises(
        ValueError,
        match="Series segment_1 contains NaNs! Set `ignore_missing=True` to drop them or impute them appropriately!",
    ):
        _ = _prepare_segment_df(df=ts_with_features.to_pandas(), segment="segment_1", ignore_missing=False)


def test_get_anomalies_isolation_forest_segment(df_segment_0):
    model = IsolationForest(n_estimators=3)
    anomalies_index = _get_anomalies_isolation_forest_segment(df_segment=df_segment_0, model=model)
    assert isinstance(anomalies_index[0], np.datetime64)


def test_get_anomalies_isolation_forest(ts_with_features):
    anomalies = get_anomalies_isolation_forest(
        ts=ts_with_features, features_to_use=["target", "exog_1"], ignore_missing=True, n_estimators=3
    )
    assert sorted(anomalies.keys()) == sorted(ts_with_features.segments)
