from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.timestamp import TimeFlagsTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import convert_ts_to_int_timestamp

INIT_PARAMS_TEMPLATE = {
    "minute_in_hour_number": False,
    "fifteen_minutes_in_hour_number": False,
    "hour_number": False,
    "half_hour_number": False,
    "half_day_number": False,
    "one_third_day_number": False,
}


@pytest.fixture
def timeflags_true_df() -> pd.DataFrame:
    """Generate dataset with answers for TimeFlagsTransform."""
    dataframes = [pd.DataFrame({"timestamp": pd.date_range("2020-06-01", "2021-06-01", freq="5T")}) for _ in range(5)]

    out_column = "timeflag"
    for i in range(len(dataframes)):
        df = dataframes[i]
        df[f"{out_column}_minute_in_hour_number"] = df["timestamp"].dt.minute.astype(int)
        df[f"{out_column}_fifteen_minutes_in_hour_number"] = (df[f"{out_column}_minute_in_hour_number"] // 15).astype(
            int
        )
        df[f"{out_column}_half_hour_number"] = (df[f"{out_column}_minute_in_hour_number"] // 30).astype(int)

        df[f"{out_column}_hour_number"] = df["timestamp"].dt.hour.astype(int)
        df[f"{out_column}_half_day_number"] = (df[f"{out_column}_hour_number"] // 12).astype(int)
        df[f"{out_column}_one_third_day_number"] = (df[f"{out_column}_hour_number"] // 8).astype(int)

        features = df.columns.difference(["timestamp"])
        df[features] = df[features].astype("category")

        df["segment"] = f"segment_{i}"
        df["target"] = 2

    flat_df = pd.concat(dataframes, ignore_index=True)
    result = TSDataset.to_dataset(flat_df)
    result.index.freq = "5T"

    return result


@pytest.fixture
def train_ts() -> TSDataset:
    """Generate dataset without timeflags"""
    dataframes = [pd.DataFrame({"timestamp": pd.date_range("2020-06-01", "2021-06-01", freq="5T")}) for _ in range(5)]

    for i in range(len(dataframes)):
        df = dataframes[i]
        df["segment"] = f"segment_{i}"
        df["target"] = 2

    flat_df = pd.concat(dataframes, ignore_index=True)
    wide_df = TSDataset.to_dataset(flat_df)

    flat_df["external_timestamp"] = flat_df["timestamp"]
    flat_df.drop(columns=["target"], inplace=True)
    df_exog = TSDataset.to_dataset(flat_df)

    ts = TSDataset(df=wide_df, df_exog=df_exog, freq="5T")
    return ts


@pytest.fixture
def train_ts_int_timestamp(train_ts) -> TSDataset:
    ts = convert_ts_to_int_timestamp(train_ts)
    return ts


@pytest.fixture
def train_ts_with_regressor(train_ts) -> TSDataset:
    df = train_ts._raw_df
    df_exog = train_ts._df_exog
    ts = TSDataset(df=df.iloc[:-10], df_exog=df_exog, freq=train_ts.freq, known_future=["external_timestamp"])
    return ts


@pytest.fixture
def train_ts_with_nans(train_ts) -> TSDataset:
    ts = train_ts
    df = ts._raw_df
    df_exog = ts._df_exog
    df_exog.loc[df_exog.index[:3], pd.IndexSlice[:, "external_timestamp"]] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq=ts.freq)
    return ts


def test_invalid_arguments_configuration():
    """Test that transform can't be created with no features to generate."""
    with pytest.raises(ValueError, match="TimeFlagsTransform feature does nothing with given init args configuration"):
        _ = TimeFlagsTransform(
            minute_in_hour_number=False,
            fifteen_minutes_in_hour_number=False,
            half_hour_number=False,
            hour_number=False,
            half_day_number=False,
            one_third_day_number=False,
        )


@pytest.mark.parametrize("in_column", [None, "external_timestamp"])
@pytest.mark.parametrize(
    "true_params",
    (
        ["minute_in_hour_number"],
        ["fifteen_minutes_in_hour_number"],
        ["hour_number"],
        ["half_hour_number"],
        ["half_day_number"],
        ["one_third_day_number"],
        [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ],
    ),
)
def test_interface_out_column(in_column: Optional[str], true_params: List[str], train_ts: TSDataset):
    """Test that transform generates correct column names using out_column parameter."""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    segments = train_ts.segments
    out_column = "timeflag"
    for key in true_params:
        init_params[key] = True
    transform = TimeFlagsTransform(**init_params, out_column=out_column, in_column=in_column)
    initial_columns = train_ts.features

    result = transform.fit_transform(train_ts).to_pandas()

    assert sorted(result.columns.names) == ["feature", "segment"]
    assert sorted(segments) == sorted(result.columns.get_level_values("segment").unique())

    true_params = [f"{out_column}_{param}" for param in true_params]
    for seg in result.columns.get_level_values("segment").unique():
        tmp_df = result[seg]
        new_columns = tmp_df.columns.difference(initial_columns)
        assert sorted(new_columns) == sorted(true_params)
        for param in true_params:
            assert tmp_df[param].dtype == "category"


@pytest.mark.parametrize("in_column", [None, "external_timestamp"])
@pytest.mark.parametrize(
    "true_params",
    (
        ["minute_in_hour_number"],
        ["fifteen_minutes_in_hour_number"],
        ["hour_number"],
        ["half_hour_number"],
        ["half_day_number"],
        ["one_third_day_number"],
        [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ],
    ),
)
def test_interface_correct_args_repr(in_column: Optional[str], true_params: List[str], train_ts: TSDataset):
    """Test that transform generates correct column names without setting out_column parameter."""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    segments = train_ts.segments
    for key in true_params:
        init_params[key] = True
    transform = TimeFlagsTransform(**init_params, in_column=in_column)
    initial_columns = train_ts.features

    result = transform.fit_transform(deepcopy(train_ts)).to_pandas()

    assert sorted(result.columns.names) == ["feature", "segment"]
    assert sorted(segments) == sorted(result.columns.get_level_values("segment").unique())

    new_columns = result.columns.get_level_values("feature").unique().difference(initial_columns)
    assert len(new_columns) == len(true_params)
    for column in new_columns:
        # check category dtype
        assert np.all(result.loc[:, pd.IndexSlice[segments, column]].dtypes == "category")

        # check that a transform can be created from column name and it generates the same results
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(deepcopy(train_ts)).to_pandas()
        new_columns_temp = df_temp.columns.get_level_values("feature").unique().difference(initial_columns)
        assert len(new_columns_temp) == 1
        generated_column = new_columns_temp[0]
        assert generated_column == column
        pd.testing.assert_frame_equal(
            df_temp.loc[:, pd.IndexSlice[segments, generated_column]], result.loc[:, pd.IndexSlice[segments, column]]
        )


@pytest.mark.parametrize(
    "in_column, ts_name",
    [
        (None, "train_ts"),
        ("external_timestamp", "train_ts"),
        ("external_timestamp", "train_ts_int_timestamp"),
    ],
)
@pytest.mark.parametrize(
    "true_params",
    (
        {"minute_in_hour_number": True},
        {"fifteen_minutes_in_hour_number": True},
        {"hour_number": True},
        {"half_hour_number": True},
        {"half_day_number": True},
        {"one_third_day_number": True},
    ),
)
def test_transform_values(
    in_column: Optional[str],
    ts_name: str,
    true_params: Dict[str, Union[bool, Tuple[int, int]]],
    timeflags_true_df: pd.DataFrame,
    request,
):
    """Test that transform generates correct values."""
    ts = request.getfixturevalue(ts_name)
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    init_params.update(true_params)
    out_column = "timeflag"
    transform = TimeFlagsTransform(**init_params, out_column=out_column, in_column=in_column)
    result = transform.fit_transform(ts).to_pandas()

    segments_true = timeflags_true_df.columns.get_level_values("segment").unique()
    segment_result = result.columns.get_level_values("segment").unique()

    assert sorted(segment_result) == sorted(segments_true)

    true_params = [f"{out_column}_{param}" for param in true_params.keys()]
    for seg in segment_result:
        segment_true = timeflags_true_df[seg]
        columns = true_params + ["target"]
        true_df = segment_true[columns].sort_index(axis=1).reset_index(drop=True)
        result_df = result[seg][columns].sort_index(axis=1).reset_index(drop=True)
        pd.testing.assert_frame_equal(true_df, result_df)


@pytest.mark.parametrize(
    "true_params",
    (
        {"minute_in_hour_number": True},
        {"fifteen_minutes_in_hour_number": True},
        {"hour_number": True},
        {"half_hour_number": True},
        {"half_day_number": True},
        {"one_third_day_number": True},
    ),
)
def test_transform_values_with_nans(true_params: Dict[str, Union[bool, Tuple[int, int]]], train_ts_with_nans):
    out_column = "timeflag"
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    init_params.update(true_params)
    transform = TimeFlagsTransform(**init_params, out_column=out_column, in_column="external_timestamp")
    result = transform.fit_transform(train_ts_with_nans).to_pandas()

    segment_result = result.columns.get_level_values("segment").unique()

    true_params = [f"{out_column}_{param}" for param in true_params.keys()]
    for seg in segment_result:
        result_df = result[seg][true_params].sort_index(axis=1).reset_index(drop=True)
        assert np.all(result_df.isna().sum() == 3)


def test_transform_index_fail_int_timestamp(train_ts_int_timestamp: TSDataset):
    transform = TimeFlagsTransform(out_column="timeflag", in_column=None)
    transform.fit(train_ts_int_timestamp)
    with pytest.raises(ValueError, match="Transform can't work with integer index, parameter in_column should be set"):
        _ = transform.transform(train_ts_int_timestamp)


@pytest.mark.parametrize(
    "true_params",
    (
        ["minute_in_hour_number"],
        ["fifteen_minutes_in_hour_number"],
        ["hour_number"],
        ["half_hour_number"],
        ["half_day_number"],
        ["one_third_day_number"],
        [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ],
    ),
)
def test_get_regressors_info_index(true_params):
    out_column = "timeflag"
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    for key in true_params:
        init_params[key] = True
    transform = TimeFlagsTransform(**init_params, out_column=out_column)

    regressors_info = transform.get_regressors_info()

    expected_regressor_info = [f"{out_column}_{param}" for param in true_params]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


def test_get_regressors_info_in_column_fail_not_fitted(train_ts):
    transform = TimeFlagsTransform(out_column="timeflag", in_column="external_timestamp")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


@pytest.mark.parametrize(
    "true_params",
    (
        ["minute_in_hour_number"],
        ["fifteen_minutes_in_hour_number"],
        ["hour_number"],
        ["half_hour_number"],
        ["half_day_number"],
        ["one_third_day_number"],
        [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ],
    ),
)
def test_get_regressors_info_in_column_fitted_exog(true_params, train_ts):
    out_column = "timeflag"
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    for key in true_params:
        init_params[key] = True
    transform = TimeFlagsTransform(**init_params, out_column=out_column, in_column="external_timestamp")

    transform.fit(train_ts)
    regressors_info = transform.get_regressors_info()

    assert regressors_info == []


@pytest.mark.parametrize(
    "true_params",
    (
        ["minute_in_hour_number"],
        ["fifteen_minutes_in_hour_number"],
        ["hour_number"],
        ["half_hour_number"],
        ["half_day_number"],
        ["one_third_day_number"],
        [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ],
    ),
)
def test_get_regressors_info_in_column_fitted_regressor(true_params, train_ts_with_regressor):
    out_column = "dateflag"
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    for key in true_params:
        init_params[key] = True
    transform = TimeFlagsTransform(**init_params, out_column=out_column, in_column="external_timestamp")

    transform.fit(train_ts_with_regressor)
    regressors_info = transform.get_regressors_info()

    expected_regressor_info = [f"{out_column}_{param}" for param in true_params]
    assert sorted(regressors_info) == sorted(expected_regressor_info)


def test_save_load(train_ts):
    ts = train_ts
    transform = TimeFlagsTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


def test_params_to_tune(train_ts):
    def skip_parameters(parameters):
        names = [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ]
        values = [not parameters[x] for x in names]
        if all(values):
            return True
        return False

    transform = TimeFlagsTransform()
    ts = train_ts
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts, skip_parameters=skip_parameters)
