from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms import LabelEncoderTransform
from etna.transforms import SegmentEncoderTransform


@pytest.fixture()
def weekly_period_df(n_repeats=15):
    segment_1 = [7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 1.0]
    segment_2 = [7.0, 7.0, 7.0, 4.0, 1.0, 7.0, 7.0]
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=n_repeats * len(segment_1)))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 * n_repeats + segment_2 * n_repeats,
            "segment": ["segment_1"] * n_repeats * len(segment_1) + ["segment_2"] * n_repeats * len(segment_2),
        }
    )
    return df


@pytest.fixture()
def ts_dataset_weekly_function_with_horizon(weekly_period_df):
    def wrapper(horizon: int) -> Tuple[TSDataset, TSDataset]:
        ts_start = sorted(set(weekly_period_df.timestamp))[-horizon]
        train, test = (
            weekly_period_df[lambda x: x.timestamp < ts_start],
            weekly_period_df[lambda x: x.timestamp >= ts_start],
        )

        ts_train = TSDataset(TSDataset.to_dataset(train), "D")
        ts_test = TSDataset(TSDataset.to_dataset(test), "D")
        return ts_train, ts_test

    return wrapper


@pytest.fixture()
def example_make_samples_df():
    target = np.arange(50).astype(float)
    timestamp = pd.date_range("2020-01-03", freq="D", periods=len(target))
    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target": target,
            "segment": ["segment_1"] * len(target),
            "regressor_float": timestamp.weekday.astype(float) * 2,
            "regressor_int": timestamp.weekday * 3,
            "regressor_bool": timestamp.weekday.isin([5, 6]),
            "regressor_str": timestamp.weekday.astype(str),
            "regressor_int_cat": timestamp.weekday.astype("category"),
        }
    )
    return df


def example_make_samples_df_int_timestamp(example_make_samples_df):
    example_make_samples_df["timestamp"] = np.arange(len(example_make_samples_df)) + 10
    return example_make_samples_df


@pytest.fixture()
def ts_different_regressors():
    df = generate_ar_df(start_time="2001-01-01", n_segments=1, periods=7)
    df_exog = generate_ar_df(start_time="2001-01-01", n_segments=1, periods=10)
    df_exog.drop(columns=["target"], inplace=True)
    df_exog["reals_exog"] = [1, 2, 3, 4, 5, 6, 7, np.NaN, np.NaN, np.NaN]
    df_exog["reals_static"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    df_exog["reals_regr"] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    df_exog["categ_exog"] = ["a", "d", "a", "d", "a", "a", "a", np.NaN, np.NaN, np.NaN]
    df_exog["categ_regr"] = ["b", "b", "e", "b", "b", "b", "b", "b", "e", "b"]
    df_exog["categ_regr_new"] = ["b", "b", "b", "b", "b", "b", "b", "c", "b", "c"]

    df_exog["categ_exog"] = df_exog["categ_exog"].fillna("Unknown")
    df = TSDataset.to_dataset(df)
    df_exog = TSDataset.to_dataset(df_exog)

    ts = TSDataset(
        df=df,
        freq="D",
        df_exog=df_exog,
        known_future=["reals_static", "reals_regr", "categ_regr", "categ_regr_new"],
    )
    return ts


@pytest.fixture()
def ts_different_regressors_encoded(ts_different_regressors):
    seg = SegmentEncoderTransform()
    label1 = LabelEncoderTransform(in_column="categ_exog", out_column="categ_exog_label", strategy="none")
    label2 = LabelEncoderTransform(in_column="categ_regr", out_column="categ_regr_label", strategy="none")
    label3 = LabelEncoderTransform(in_column="categ_regr_new", out_column="categ_regr_new_label", strategy="none")

    seg.fit_transform(ts_different_regressors)
    label1.fit_transform(ts_different_regressors)
    label2.fit_transform(ts_different_regressors)
    label3.fit_transform(ts_different_regressors)

    return ts_different_regressors
