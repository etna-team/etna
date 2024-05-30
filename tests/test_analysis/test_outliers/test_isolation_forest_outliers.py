import pytest

from etna.analysis.outliers.isolation_forest_outliers import _select_features
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def ts_with_features():
    df = generate_ar_df(n_segments=3, periods=10, start_time="2000-01-01")
    df["exog_1"] = [1] * 10 + [2] * 10 + [3] * 10
    ts = TSDataset(df=df.drop(columns=["exog_1"]), freq="D", df_exog=df.drop(columns=["target"]))
    return ts


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
