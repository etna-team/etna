import pathlib
import tempfile
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from lightning_fabric.utilities.seed import seed_everything

from etna.datasets import TSDataset
from etna.datasets.utils import timestamp_range
from etna.pipeline.base import AbstractPipeline


def get_loaded_pipeline(pipeline: AbstractPipeline, ts: TSDataset = None) -> AbstractPipeline:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        pipeline.save(path)
        if ts is None:
            loaded_pipeline = pipeline.load(path)
        else:
            loaded_pipeline = pipeline.load(path, ts=ts)
    return loaded_pipeline


def assert_pipeline_equals_loaded_original(
    pipeline: AbstractPipeline, ts: TSDataset, load_ts: bool = True
) -> Tuple[AbstractPipeline, AbstractPipeline]:

    initial_ts = deepcopy(ts)

    pipeline.fit(ts)
    seed_everything(0)
    forecast_ts_1 = pipeline.forecast()

    if load_ts:
        loaded_pipeline = get_loaded_pipeline(pipeline, ts=initial_ts)
        seed_everything(0)
        forecast_ts_2 = loaded_pipeline.forecast()
    else:
        loaded_pipeline = get_loaded_pipeline(pipeline)
        seed_everything(0)
        forecast_ts_2 = loaded_pipeline.forecast(ts=initial_ts)

    pd.testing.assert_frame_equal(forecast_ts_1.to_pandas(), forecast_ts_2.to_pandas())

    return pipeline, loaded_pipeline


def assert_pipeline_forecast_raise_error_if_no_ts(pipeline: AbstractPipeline, ts: TSDataset):
    with pytest.raises(ValueError, match="There is no ts to forecast!"):
        _ = pipeline.forecast()

    pipeline.fit(ts, save_ts=False)
    with pytest.raises(ValueError, match="There is no ts to forecast!"):
        _ = pipeline.forecast()


def assert_pipeline_forecasts_without_self_ts(
    pipeline: AbstractPipeline, ts: TSDataset, horizon: int
) -> AbstractPipeline:
    pipeline.fit(ts=ts, save_ts=False)
    forecast_ts = pipeline.forecast(ts=ts)
    forecast_df = forecast_ts.to_pandas(flatten=True)

    if ts.has_hierarchy():
        expected_segments = ts.hierarchical_structure.get_level_segments(forecast_ts.current_df_level)
    else:
        expected_segments = ts.segments
    assert forecast_ts.segments == expected_segments

    expected_index = timestamp_range(start=ts.timestamps[-1], periods=horizon + 1, freq=ts.freq)[1:]
    expected_index.name = "timestamp"
    pd.testing.assert_index_equal(forecast_ts.timestamps, expected_index)
    assert not forecast_df["target"].isna().any()

    return pipeline


def assert_pipeline_forecasts_given_ts(pipeline: AbstractPipeline, ts: TSDataset, horizon: int) -> AbstractPipeline:
    fit_ts = deepcopy(ts)
    fit_ts._df = fit_ts._df.iloc[:-horizon]
    to_forecast_ts = deepcopy(ts)

    pipeline.fit(ts=fit_ts)
    forecast_ts = pipeline.forecast(ts=to_forecast_ts)
    forecast_df = forecast_ts.to_pandas(flatten=True)

    if ts.has_hierarchy():
        expected_segments = ts.hierarchical_structure.get_level_segments(forecast_ts.current_df_level)
    else:
        expected_segments = to_forecast_ts.segments
    assert forecast_ts.segments == expected_segments

    expected_index = timestamp_range(start=to_forecast_ts.timestamps[-1], periods=horizon + 1, freq=ts.freq)[1:]
    expected_index.name = "timestamp"
    pd.testing.assert_index_equal(forecast_ts.timestamps, expected_index)
    assert not forecast_df["target"].isna().any()

    return pipeline


def assert_pipeline_forecasts_given_ts_with_prediction_intervals(
    pipeline: AbstractPipeline, ts: TSDataset, horizon: int, **forecast_params
) -> AbstractPipeline:
    fit_ts = deepcopy(ts)
    fit_ts._df = fit_ts._df.iloc[:-horizon]
    to_forecast_ts = deepcopy(ts)

    pipeline.fit(fit_ts)
    forecast_ts = pipeline.forecast(
        ts=to_forecast_ts, prediction_interval=True, quantiles=[0.025, 0.975], **forecast_params
    )
    forecast_df = forecast_ts.to_pandas(flatten=True)

    if ts.has_hierarchy():
        expected_segments = ts.hierarchical_structure.get_level_segments(forecast_ts.current_df_level)
    else:
        expected_segments = to_forecast_ts.segments
    assert forecast_ts.segments == expected_segments

    expected_index = timestamp_range(start=to_forecast_ts.timestamps[-1], periods=horizon + 1, freq=ts.freq)[1:]
    expected_index.name = "timestamp"
    pd.testing.assert_index_equal(forecast_ts.timestamps, expected_index)

    assert not forecast_df["target"].isna().any()
    assert not forecast_df["target_0.025"].isna().any()
    assert not forecast_df["target_0.975"].isna().any()

    return pipeline


def assert_pipeline_predicts(
    pipeline: AbstractPipeline, ts: TSDataset, start_idx: int, end_idx: int
) -> AbstractPipeline:
    predict_ts = deepcopy(ts)
    pipeline.fit(ts)

    start_timestamp = ts.timestamps[start_idx]
    end_timestamp = ts.timestamps[end_idx]
    num_points = end_idx - start_idx + 1

    predict_ts = pipeline.predict(ts=predict_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    predict_df = predict_ts.to_pandas(flatten=True)

    if ts.has_hierarchy():
        expected_segments = ts.hierarchical_structure.get_level_segments(predict_ts.current_df_level)
    else:
        expected_segments = predict_ts.segments
    assert predict_ts.segments == expected_segments

    expected_index = timestamp_range(start=start_timestamp, periods=num_points, freq=ts.freq)
    expected_index.name = "timestamp"
    pd.testing.assert_index_equal(predict_ts.timestamps, expected_index)

    assert not np.any(predict_df["target"].isna())

    return pipeline
