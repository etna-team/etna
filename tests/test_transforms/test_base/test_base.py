from typing import List
from unittest.mock import Mock

import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms import AddConstTransform
from etna.transforms import IrreversibleTransform
from etna.transforms import ReversibleTransform


class TransformMock(IrreversibleTransform):
    def get_regressors_info(self) -> List[str]:
        return ["regressor_test"]

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class ReversibleTransformMock(ReversibleTransform):
    def get_regressors_info(self) -> List[str]:
        return ["regressor_test"]

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class SimpleAddColumnTransform(IrreversibleTransform):
    def get_regressors_info(self) -> List[str]:
        return ["regressor_test"]

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = df.rename(columns={"target": "regressor_test"}, level=1)
        return pd.concat([df, feat], axis=1)


@pytest.fixture
def remove_features_df():
    df = generate_ar_df(periods=10, n_segments=3, start_time="2000-01-01")
    df["exog_1"] = 1
    df["target_0.01"] = 2
    df = TSDataset.to_dataset(df)

    df_transformed = generate_ar_df(periods=10, n_segments=3, start_time="2000-01-01")
    df_transformed = TSDataset.to_dataset(df_transformed)
    return df, df_transformed


@pytest.fixture()
def remove_columns_ts(remove_features_df):
    df, df_transformed = remove_features_df

    df_exog = df.loc[:, pd.IndexSlice[:, "exog_1"]]
    intervals = df.loc[:, pd.IndexSlice[:, "target_0.01"]]
    df = df.drop(columns=["exog_1", "target_0.01"], level="feature")
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    ts.add_prediction_intervals(prediction_intervals_df=intervals)

    ts_transformed = TSDataset(df=df_transformed, freq="D")
    return ts, ts_transformed


@pytest.mark.parametrize(
    "required_features, expected_features",
    [("all", "all"), (["target", "segment"], ["target", "segment"])],
)
def test_required_features(required_features, expected_features):
    transform = TransformMock(required_features=required_features)
    assert transform.required_features == expected_features


def test_update_dataset_remove_features(remove_features_df):
    df, df_transformed = remove_features_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.drop_features = Mock()
    expected_features_to_remove = list(
        set(df.columns.get_level_values("feature")) - set(df_transformed.columns.get_level_values("feature"))
    )
    transform = TransformMock(required_features=["target"])

    transform._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
    ts.drop_features.assert_called_with(features=expected_features_to_remove, drop_from_exog=False)


def test_update_dataset_update_features(remove_features_df):
    df, df_transformed = remove_features_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.update_features_from_pandas = Mock()
    transform = TransformMock(required_features=["target"])

    transform._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
    ts.update_features_from_pandas.assert_called()


def test_update_dataset_add_features(remove_features_df):
    df_transformed, df = remove_features_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.add_features_from_pandas = Mock()
    transform = TransformMock(required_features=["target"])

    transform._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
    ts.add_features_from_pandas.assert_called()


@pytest.mark.parametrize(
    "required_features",
    [("all"), (["target", "segment"])],
)
def test_fit_request_correct_columns(required_features):
    ts = Mock()
    transform = TransformMock(required_features=required_features)

    transform.fit(ts=ts)
    ts.to_pandas.assert_called_with(flatten=False, features=required_features)


@pytest.mark.parametrize(
    "required_features",
    [("all"), (["target", "segment"])],
)
def test_transform_request_correct_columns(remove_columns_ts, required_features):
    ts, _ = remove_columns_ts
    ts.to_pandas = Mock(return_value=ts._df)

    transform = TransformMock(required_features=required_features)
    transform._update_dataset = Mock()

    transform.transform(ts=ts)
    ts.to_pandas.assert_called_with(flatten=False, features=required_features)


@pytest.mark.parametrize(
    "required_features",
    [("all"), (["target", "segment"])],
)
def test_transform_request_update_dataset(remove_columns_ts, required_features):
    ts, _ = remove_columns_ts
    columns_before = set(ts.features)
    ts.to_pandas = Mock(return_value=ts._df)

    transform = TransformMock(required_features=required_features)
    transform._update_dataset = Mock()

    transform.transform(ts=ts)
    transform._update_dataset.assert_called_with(ts=ts, columns_before=columns_before, df_transformed=ts._df)


@pytest.mark.parametrize(
    "in_column, expected_required_features",
    [(["target"], ["target", "target_0.01"]), (["exog_1"], ["exog_1"])],
)
def test_inverse_transform_add_target_quantiles(remove_columns_ts, in_column, expected_required_features):
    ts, _ = remove_columns_ts

    transform = ReversibleTransformMock(required_features=in_column)
    required_features = transform._get_inverse_transform_required_features(ts)
    assert sorted(required_features) == sorted(expected_required_features)


def test_inverse_transform_request_update_dataset(remove_columns_ts):
    ts, _ = remove_columns_ts
    columns_before = set(ts.features)
    ts.to_pandas = Mock(return_value=ts._df)

    transform = ReversibleTransformMock(required_features="all")
    transform._inverse_transform = Mock()
    transform._update_dataset = Mock()

    transform.inverse_transform(ts=ts)
    expected_df_transformed = transform._inverse_transform.return_value
    transform._update_dataset.assert_called_with(
        ts=ts, columns_before=columns_before, df_transformed=expected_df_transformed
    )


def test_double_apply_add_columns_transform(remove_features_df):
    df, _ = remove_features_df
    ts = TSDataset(df=df, freq="D")

    transform = SimpleAddColumnTransform(required_features=["target"])
    ts_transformed = transform.fit_transform(ts=ts)
    ts_transformed = transform.transform(ts=ts_transformed)
    assert (
        ts_transformed._df.columns.get_level_values("feature").tolist()
        == ["exog_1", "regressor_test", "target", "target_0.01"] * 3
    )


@pytest.fixture
def ts_with_target_components():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target": 3,
            "exog": 10,
            "segment": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target": 7,
            "exog": 10,
            "segment": 2,
        }
    )

    components_df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 1,
            "target_component_b": 2,
            "segment": 1,
        }
    )
    components_df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 3,
            "target_component_b": 4,
            "segment": 2,
        }
    )

    df = pd.concat([df_1, df_2])
    components_df = pd.concat([components_df_1, components_df_2])

    df = TSDataset.to_dataset(df=df)
    components_df = TSDataset.to_dataset(df=components_df)

    ts = TSDataset(df=df, freq="D")
    ts.add_target_components(target_components_df=components_df)
    return ts


@pytest.fixture
def inverse_transformed_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 1 * (3 + 10) / 3,
            "target_component_b": 2 * (3 + 10) / 3,
            "segment": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 3 * (7 + 10) / 7,
            "target_component_b": 4 * (7 + 10) / 7,
            "segment": 2,
        }
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    df.index.freq = "D"
    return df


def test_inverse_transform_with_target_components(ts_with_target_components, inverse_transformed_components_df):
    transform = AddConstTransform(in_column="target", value=-10)
    transform.inverse_transform(ts=ts_with_target_components)
    pd.testing.assert_frame_equal(ts_with_target_components.get_target_components(), inverse_transformed_components_df)


def test_inverse_transform_with_target_components_target_not_in_required_features(ts_with_target_components):
    target_components_before = ts_with_target_components.get_target_components()
    transform = AddConstTransform(in_column="exog", value=-10)
    transform.inverse_transform(ts=ts_with_target_components)
    pd.testing.assert_frame_equal(ts_with_target_components.get_target_components(), target_components_before)


@pytest.mark.parametrize(
    "transform", [TransformMock(required_features="all"), ReversibleTransformMock(required_features="all")]
)
def test_default_params_to_tune(transform):
    assert transform.params_to_tune() == {}
