import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_iqr
from etna.analysis import get_anomalies_isolation_forest
from etna.analysis import get_anomalies_mad
from etna.analysis import get_anomalies_median
from etna.analysis import get_anomalies_prediction_interval
from etna.datasets.tsdataset import TSDataset
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import FourierTransform
from etna.transforms import HolidayTransform
from etna.transforms import LagTransform
from etna.transforms import MedianOutliersTransform
from etna.transforms import PredictionIntervalOutliersTransform
from etna.transforms.outliers import IForestOutlierTransform
from etna.transforms.outliers import IQROutlierTransform
from etna.transforms.outliers import MADOutlierTransform
from tests.test_transforms.utils import assert_column_changes
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.test_transforms.utils import find_columns_diff
from tests.utils import select_segments_subset


def insert_column(ts, info_col, timestamp, segment):
    return ts.add_columns_from_pandas(
        TSDataset.to_dataset(
            pd.DataFrame(
                {
                    "is_holiday": info_col,
                    "timestamp": timestamp,
                    "segment": segment,
                }
            )
        )
    )


def made_specific_ds(ts, add_error=True):
    timestamp = pd.date_range("2021-01-01", end="2021-02-20", freq="D")
    info_col1 = [1 if np.sin(i) > 0.5 else 0 for i in range(len(timestamp))]
    info_col2 = [1 if np.sin(i) > 0 else 0 for i in range(len(timestamp))]

    if add_error:
        info_col1[9] = 4
        info_col2[10] = 14

    insert_column(ts, info_col1, timestamp, "1")
    insert_column(ts, info_col2, timestamp, "2")

    return ts


def compare_outputs(ts, in_column, method, transform_constructor, method_kwargs, constructor_kwargs):
    detection_method_results = method(ts, in_column=in_column, **method_kwargs)
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)

    # save for each segment index without existing nans
    non_nan_index = {}
    for segment in ts.segments:
        non_nan_index[segment] = ts[:, segment, in_column].dropna().index
    # convert to df to ignore different lengths of series
    transformed_df = transform.fit_transform(ts).to_pandas()
    for segment in ts.segments:
        nan_timestamps = detection_method_results[segment]
        transformed_column = transformed_df.loc[non_nan_index[segment], pd.IndexSlice[segment, in_column]]
        assert np.all(transformed_column[transformed_column.isna()].index == nan_timestamps)


@pytest.fixture()
def outliers_solid_tsds():
    """Create TSDataset with outliers and same last date."""
    timestamp = pd.date_range("2021-01-01", end="2021-02-20", freq="D")
    target1 = [np.sin(i) for i in range(len(timestamp))]
    target1[10] += 10

    target2 = [np.sin(i) for i in range(len(timestamp))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12

    df1 = pd.DataFrame({"timestamp": timestamp, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": target2, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df_exog = df.copy()
    df_exog.columns = ["timestamp", "regressor_1", "segment"]
    ts = TSDataset(
        df=TSDataset.to_dataset(df).iloc[:-10],
        df_exog=TSDataset.to_dataset(df_exog),
        freq="D",
        known_future="all",
    )
    return ts


@pytest.fixture()
def outliers_solid_tsds_with_holidays(outliers_solid_tsds):
    """Create TSDataset with outliers with holidays"""
    ts = outliers_solid_tsds
    holiday_transform = HolidayTransform(iso_code="RUS", mode="binary", out_column="is_holiday")
    ts = holiday_transform.fit_transform(ts)
    return ts


@pytest.fixture()
def outliers_solid_tsds_with_error(outliers_solid_tsds):
    """Create TSDataset with outliers error inside ts, incorrect type column"""
    return made_specific_ds(outliers_solid_tsds, add_error=True)


@pytest.fixture()
def outliers_solid_tsds_non_regressor_holiday(outliers_solid_tsds):
    """Create TSDataset with outliers inside ts non regressor"""
    return made_specific_ds(outliers_solid_tsds, add_error=False)


@pytest.mark.parametrize("attribute_name,value_type", (("outliers_timestamps", list), ("original_values", pd.Series)))
def test_density_outliers_deprecated_store_attributes(outliers_solid_tsds, attribute_name, value_type):
    transform = DensityOutliersTransform(in_column="target")
    transform.fit(ts=outliers_solid_tsds)

    with pytest.warns(DeprecationWarning, match=".* is deprecated and will be removed"):
        res = getattr(transform, attribute_name)

    assert isinstance(res, dict)
    for key, value in res.items():
        assert isinstance(key, str)
        assert isinstance(value, value_type)


@pytest.mark.parametrize("in_column", ["target", "regressor_1"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs",
    [
        (MedianOutliersTransform, {}),
        (DensityOutliersTransform, {}),
        (PredictionIntervalOutliersTransform, dict(model=ProphetModel)),
        (IForestOutlierTransform, {}),
        (IQROutlierTransform, {}),
        (MADOutlierTransform, {}),
    ],
)
def test_interface(transform_constructor, constructor_kwargs, outliers_solid_tsds: TSDataset, in_column):
    """Checks outliers transforms doesn't change structure of dataframe."""
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)
    start_columns = outliers_solid_tsds.columns
    outliers_solid_tsds = transform.fit_transform(ts=outliers_solid_tsds)
    assert np.all(start_columns == outliers_solid_tsds.columns)


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs, method, method_kwargs",
    [
        (MedianOutliersTransform, {}, get_anomalies_median, {}),
        (DensityOutliersTransform, {}, get_anomalies_density, {}),
        (
            PredictionIntervalOutliersTransform,
            dict(model=ProphetModel),
            get_anomalies_prediction_interval,
            dict(model=ProphetModel),
        ),
        (
            IForestOutlierTransform,
            {"ignore_missing": True, "random_state": 42},
            get_anomalies_isolation_forest,
            {"ignore_missing": True, "random_state": 42},
        ),
    ],
)
def test_outliers_detection(transform_constructor, constructor_kwargs, method, outliers_tsds, method_kwargs, in_column):
    """Checks that outliers transforms detect anomalies according to methods from etna.analysis."""
    compare_outputs(
        ts=outliers_tsds,
        in_column=in_column,
        method=method,
        transform_constructor=transform_constructor,
        method_kwargs=method_kwargs,
        constructor_kwargs=constructor_kwargs,
    )


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs, method, method_kwargs",
    [
        (
            IForestOutlierTransform,
            {"ignore_missing": False, "random_state": 42},
            get_anomalies_isolation_forest,
            {"ignore_missing": False, "random_state": 42},
        ),
        (
            IQROutlierTransform,
            {"iqr_scale": 3, "window_size": 15},
            get_anomalies_iqr,
            {"iqr_scale": 3, "window_size": 15},
        ),
        (
            MADOutlierTransform,
            {"mad_scale": 3, "window_size": 20},
            get_anomalies_mad,
            {"mad_scale": 3, "window_size": 20},
        ),
    ],
)
def test_outliers_detection_without_missing(
    transform_constructor, constructor_kwargs, method, outliers_tsds_without_missing, method_kwargs, in_column
):
    """Checks that outliers transforms detect anomalies according to methods from etna.analysis."""
    compare_outputs(
        ts=outliers_tsds_without_missing,
        in_column=in_column,
        method=method,
        transform_constructor=transform_constructor,
        method_kwargs=method_kwargs,
        constructor_kwargs=constructor_kwargs,
    )


@pytest.mark.parametrize("in_column", ["target", "regressor_1"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs",
    [
        (MedianOutliersTransform, {}),
        (DensityOutliersTransform, {}),
        (PredictionIntervalOutliersTransform, dict(model=ProphetModel)),
        (IForestOutlierTransform, {}),
        (IQROutlierTransform, {}),
        (MADOutlierTransform, {}),
    ],
)
def test_inverse_transform_train(transform_constructor, constructor_kwargs, outliers_solid_tsds, in_column):
    """Checks that inverse transform returns dataset to its original form."""
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)
    original_df = outliers_solid_tsds.to_pandas()
    outliers_solid_tsds = transform.fit_transform(ts=outliers_solid_tsds)
    transform.inverse_transform(ts=outliers_solid_tsds)

    assert np.all(original_df == outliers_solid_tsds.df)


@pytest.mark.parametrize("in_column", ["target", "regressor_1"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs",
    [
        (MedianOutliersTransform, {}),
        (DensityOutliersTransform, {}),
        (PredictionIntervalOutliersTransform, dict(model=ProphetModel)),
        (IForestOutlierTransform, {}),
        (IQROutlierTransform, {}),
        (MADOutlierTransform, {}),
    ],
)
def test_inverse_transform_future(transform_constructor, constructor_kwargs, outliers_solid_tsds, in_column):
    """Checks that inverse transform does not change the future."""
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)
    outliers_solid_tsds.fit_transform([transform])
    future = outliers_solid_tsds.make_future(future_steps=10, transforms=[transform])
    original_future_df = future.to_pandas()
    future.inverse_transform([transform])
    # check equals and has nans in the same places
    assert np.all((future.df == original_future_df) | (future.df.isna() & original_future_df.isna()))


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
        IForestOutlierTransform(in_column="target"),
        IQROutlierTransform(in_column="target"),
        MADOutlierTransform(in_column="target"),
    ),
)
def test_transform_raise_error_if_not_fitted(transform, outliers_solid_tsds):
    """Test that transform for one segment raise error when calling transform without being fit."""
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(ts=outliers_solid_tsds)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
        IForestOutlierTransform(in_column="target"),
        IQROutlierTransform(in_column="target"),
        MADOutlierTransform(in_column="target"),
    ),
)
def test_inverse_transform_raise_error_if_not_fitted(transform, outliers_solid_tsds):
    """Test that transform for one segment raise error when calling inverse_transform without being fit."""
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.inverse_transform(ts=outliers_solid_tsds)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
        IForestOutlierTransform(in_column="target"),
        IQROutlierTransform(in_column="target"),
        MADOutlierTransform(in_column="target"),
    ),
)
def test_transform_new_segments_fail(transform, outliers_solid_tsds):
    train_ts = select_segments_subset(ts=outliers_solid_tsds, segments=["1"])
    test_ts = select_segments_subset(ts=outliers_solid_tsds, segments=["2"])

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_ts)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
        IForestOutlierTransform(in_column="target"),
        IQROutlierTransform(in_column="target"),
        MADOutlierTransform(in_column="target"),
    ),
)
def test_inverse_transform_new_segments_fail(transform, outliers_solid_tsds):
    train_ts = select_segments_subset(ts=outliers_solid_tsds, segments=["1"])
    test_ts = select_segments_subset(ts=outliers_solid_tsds, segments=["2"])

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.inverse_transform(test_ts)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
        IForestOutlierTransform(in_column="target", ignore_missing=True),
    ),
)
def test_fit_transform_with_nans(transform, ts_diff_endings):
    _ = transform.fit_transform(ts_diff_endings)


@pytest.mark.parametrize(
    "transform",
    (IForestOutlierTransform(in_column="target", ignore_missing=False), IQROutlierTransform(in_column="target")),
)
def test_fit_transform_with_nans_error(transform, ts_diff_endings):
    with pytest.raises(ValueError):
        _ = transform.fit_transform(ts_diff_endings)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        IForestOutlierTransform(in_column="target"),
        IQROutlierTransform(in_column="target"),
        MADOutlierTransform(in_column="target"),
    ),
)
def test_save_load(transform, outliers_solid_tsds):
    assert_transformation_equals_loaded_original(transform=transform, ts=outliers_solid_tsds)


@pytest.mark.parametrize(
    "transform",
    (
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
        PredictionIntervalOutliersTransform(in_column="target", model=SARIMAXModel),
    ),
)
def test_save_load_prediction_interval(transform, outliers_solid_tsds):
    assert_transformation_equals_loaded_original(transform=transform, ts=outliers_solid_tsds)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model="sarimax"),
        IForestOutlierTransform(in_column="target"),
        IQROutlierTransform(in_column="target"),
        MADOutlierTransform(in_column="target"),
    ),
)
def test_params_to_tune(transform, outliers_solid_tsds):
    ts = outliers_solid_tsds
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target", ignore_flag_column="is_holiday"),
        DensityOutliersTransform(in_column="target", ignore_flag_column="is_holiday"),
        PredictionIntervalOutliersTransform(in_column="target", model="sarimax", ignore_flag_column="is_holiday"),
        IForestOutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
        IQROutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
        MADOutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
    ),
)
def test_correct_ignore_flag(transform, outliers_solid_tsds_with_holidays):
    ts = outliers_solid_tsds_with_holidays
    transform.fit(ts)
    ts_output = transform.transform(ts)
    assert not any(ts_output["2021-01-06":"2021-01-06", "1", "target"].isna())


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target", ignore_flag_column="is_holiday"),
        DensityOutliersTransform(in_column="target", ignore_flag_column="is_holiday"),
        PredictionIntervalOutliersTransform(in_column="target", model="sarimax", ignore_flag_column="is_holiday"),
        IForestOutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
        IQROutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
        MADOutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
    ),
)
def test_incorrect_not_exists_column(transform, outliers_solid_tsds):
    ts = outliers_solid_tsds
    with pytest.raises(ValueError, match='Name ignore_flag_column="is_holiday" not find.'):
        transform.fit(ts)
        _ = transform.transform(ts)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target", ignore_flag_column="is_holiday"),
        DensityOutliersTransform(in_column="target", ignore_flag_column="is_holiday"),
        PredictionIntervalOutliersTransform(in_column="target", model="sarimax", ignore_flag_column="is_holiday"),
        IForestOutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
        IQROutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
        MADOutlierTransform(in_column="target", ignore_flag_column="is_holiday"),
    ),
)
def test_incorrect_type_ignore_flag(transform, outliers_solid_tsds_with_error):
    ts = outliers_solid_tsds_with_error
    with pytest.raises(
        ValueError,
        match=re.escape("Columns ignore_flag contain non binary value: columns: \"is_holiday\" in segment: ['1', '2']"),
    ):
        transform.fit(ts)
        _ = transform.transform(ts)


@pytest.mark.parametrize(
    "transform, expected_changes",
    [
        (MedianOutliersTransform(in_column="target", ignore_flag_column="is_holiday"), {"change": {"target"}}),
        (DensityOutliersTransform(in_column="target", ignore_flag_column="is_holiday"), {"change": {"target"}}),
        (
            PredictionIntervalOutliersTransform(in_column="target", model="sarimax", ignore_flag_column="is_holiday"),
            {"change": {"target"}},
        ),
        (IForestOutlierTransform(in_column="target", ignore_flag_column="is_holiday"), {"change": {"target"}}),
        (IQROutlierTransform(in_column="target", ignore_flag_column="is_holiday"), {"change": {"target"}}),
        (MADOutlierTransform(in_column="target", ignore_flag_column="is_holiday"), {"change": {"target"}}),
    ],
)
def test_full_train_with_outliers(transform, expected_changes, outliers_solid_tsds_with_holidays):
    ts = outliers_solid_tsds_with_holidays

    train_ts = deepcopy(ts)
    test_ts = deepcopy(ts)

    transform.fit(train_ts)

    transformed_test_ts = transform.transform(deepcopy(test_ts))

    inverse_transformed_test_ts = transform.inverse_transform(deepcopy(transformed_test_ts))

    # check
    assert_column_changes(ts_1=transformed_test_ts, ts_2=inverse_transformed_test_ts, expected_changes=expected_changes)
    flat_test_df = test_ts.to_pandas(flatten=True)
    flat_transformed_test_df = transformed_test_ts.to_pandas(flatten=True)
    flat_inverse_transformed_test_df = inverse_transformed_test_ts.to_pandas(flatten=True)
    created_columns, removed_columns, changed_columns = find_columns_diff(
        flat_transformed_test_df, flat_inverse_transformed_test_df
    )
    pd.testing.assert_frame_equal(
        flat_test_df[list(changed_columns)], flat_inverse_transformed_test_df[list(changed_columns)]
    )


@pytest.mark.parametrize(
    "transform",
    [
        (MedianOutliersTransform(in_column="target", ignore_flag_column="is_holiday")),
        (DensityOutliersTransform(in_column="target", ignore_flag_column="is_holiday")),
        (PredictionIntervalOutliersTransform(in_column="target", model="sarimax", ignore_flag_column="is_holiday")),
        (IForestOutlierTransform(in_column="target", ignore_flag_column="is_holiday")),
        (IQROutlierTransform(in_column="target", ignore_flag_column="is_holiday")),
        (MADOutlierTransform(in_column="target", ignore_flag_column="is_holiday")),
    ],
)
def test_full_pipeline(transform, outliers_solid_tsds):
    ts = outliers_solid_tsds

    holiday_transform = HolidayTransform(iso_code="RUS", mode="binary", out_column="is_holiday")
    pipeline = Pipeline(NaiveModel(lag=1), transforms=[holiday_transform, transform], horizon=3)
    pipeline.fit(ts)


@pytest.mark.parametrize(
    "transform",
    [
        (MedianOutliersTransform(in_column="target", ignore_flag_column="is_holiday")),
        (DensityOutliersTransform(in_column="target", ignore_flag_column="is_holiday")),
        (PredictionIntervalOutliersTransform(in_column="target", model="sarimax", ignore_flag_column="is_holiday")),
        (IForestOutlierTransform(in_column="target", ignore_flag_column="is_holiday")),
        (IQROutlierTransform(in_column="target", ignore_flag_column="is_holiday")),
        (MADOutlierTransform(in_column="target", ignore_flag_column="is_holiday")),
    ],
)
def test_advance_usage_data_in_transform_nonregressor(transform, outliers_solid_tsds_non_regressor_holiday):
    ts = outliers_solid_tsds_non_regressor_holiday
    transform.fit(ts)
    _ = transform.transform(ts)


@pytest.mark.parametrize(
    "features",
    [
        (
            [
                HolidayTransform(iso_code="RUS", mode="binary", out_column="is_holiday"),
                LagTransform(in_column="target", lags=[7, 8, 9], out_column="lag"),
            ]
        ),
        ([DateFlagsTransform(out_column="df"), FourierTransform(period=30, order=4, out_column="fourier")]),
    ],
)
def test_isolation_forest_with_features_from_transforms(outliers_solid_tsds, features):
    ts = outliers_solid_tsds
    anomaly_detection = IForestOutlierTransform(in_column="target")
    pipeline = Pipeline(NaiveModel(lag=1), transforms=features + [anomaly_detection], horizon=3)
    pipeline.fit(ts)
