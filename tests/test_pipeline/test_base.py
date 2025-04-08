import pathlib
import re
from typing import Any
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.pipeline import FoldMask
from etna.pipeline.base import BasePipeline


@pytest.mark.parametrize(
    "first_train_timestamp, last_train_timestamp, target_timestamps",
    [
        (None, pd.Timestamp("2020-01-05"), [pd.Timestamp("2020-01-06")]),
        (None, pd.Timestamp("2020-01-05"), ["2020-01-06"]),
        (None, "2020-01-05", [pd.Timestamp("2020-01-06")]),
        (None, "2020-01-05", ["2020-01-06"]),
        (None, "2020-01-05", ["2020-01-06", "2020-01-07"]),
        (None, "2020-01-05", ["2020-01-07", "2020-01-06"]),
        (None, 5, [6]),
        (None, 5, [6, 7]),
        (None, 5, [7, 6]),
        ("2020-01-01", "2020-01-01", ["2020-01-06"]),
        ("2020-01-01", "2020-01-05", ["2020-01-06"]),
        ("2020-01-01", "2020-01-05", ["2020-01-06", "2020-01-07"]),
        (1, 1, [6]),
        (1, 5, [6]),
        (1, 5, [6, 7]),
    ],
)
def test_fold_mask_init_ok(first_train_timestamp, last_train_timestamp, target_timestamps):
    _ = FoldMask(
        first_train_timestamp=first_train_timestamp,
        last_train_timestamp=last_train_timestamp,
        target_timestamps=target_timestamps,
    )


@pytest.mark.parametrize(
    "first_train_timestamp, last_train_timestamp, target_timestamps",
    [
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-01"), [pd.Timestamp("2020-01-02")]),
        (5, 1, [2]),
        ("2020-01-05", "2020-01-01", ["2020-01-02"]),
    ],
)
def test_fold_mask_init_fail_wrong_order_first_last_training_timestamps(
    first_train_timestamp, last_train_timestamp, target_timestamps
):
    with pytest.raises(ValueError, match="Last train timestamp should be not sooner than first train timestamp"):
        _ = FoldMask(
            first_train_timestamp=first_train_timestamp,
            last_train_timestamp=last_train_timestamp,
            target_timestamps=target_timestamps,
        )


@pytest.mark.parametrize(
    "first_train_timestamp, last_train_timestamp, target_timestamps",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"), []),
        (None, pd.Timestamp("2020-01-05"), []),
        (1, 5, []),
        ("2020-01-01", "2020-01-05", []),
    ],
)
def test_fold_mask_init_fail_wrong_fail_empty_target_timestamps(
    first_train_timestamp, last_train_timestamp, target_timestamps
):
    with pytest.raises(ValueError, match="Target timestamps shouldn't be empty"):
        _ = FoldMask(
            first_train_timestamp=first_train_timestamp,
            last_train_timestamp=last_train_timestamp,
            target_timestamps=target_timestamps,
        )


@pytest.mark.parametrize(
    "first_train_timestamp, last_train_timestamp, target_timestamps",
    [
        (
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-05"),
            [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-06")],
        ),
        (None, pd.Timestamp("2020-01-05"), [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-06")]),
        (1, 5, [6, 6]),
        ("2020-01-01", "2020-01-05", ["2020-01-06", "2020-01-06"]),
    ],
)
def test_fold_mask_init_fail_wrong_fail_duplicated_target_timestamps(
    first_train_timestamp, last_train_timestamp, target_timestamps
):
    with pytest.raises(ValueError, match="Target timestamps shouldn't contain duplicates"):
        _ = FoldMask(
            first_train_timestamp=first_train_timestamp,
            last_train_timestamp=last_train_timestamp,
            target_timestamps=target_timestamps,
        )


@pytest.mark.parametrize(
    "first_train_timestamp, last_train_timestamp, target_timestamps",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"), [pd.Timestamp("2020-01-02")]),
        (None, pd.Timestamp("2020-01-05"), [pd.Timestamp("2020-01-02")]),
        (1, 5, [2]),
        ("2020-01-01", "2020-01-05", ["2020-01-02"]),
    ],
)
def test_fold_mask_init_fail_target_timestamps_before_train_end(
    first_train_timestamp, last_train_timestamp, target_timestamps
):
    with pytest.raises(ValueError, match="Target timestamps should be strictly later then last train timestamp"):
        _ = FoldMask(
            first_train_timestamp=first_train_timestamp,
            last_train_timestamp=last_train_timestamp,
            target_timestamps=target_timestamps,
        )


@pytest.mark.parametrize(
    "first_train_timestamp, last_train_timestamp, target_timestamps",
    [
        (None, 5, [pd.Timestamp("2020-01-06")]),
        (None, 5, ["2020-01-06"]),
        (None, "2020-01-05", [6]),
        (None, 5, [6, "2020-01-07"]),
        (None, "2020-01-05", ["2020-01-06", 7]),
        (1, 5, ["2020-01-06"]),
        (1, "2020-01-05", [6]),
        (1, "2020-01-05", ["2020-01-06"]),
        ("2020-01-01", 5, [6]),
        ("2020-01-01", "2020-01-05", [6]),
        ("2020-01-01", 5, ["2020-01-06"]),
        (1, 5, [6, "2020-01-07"]),
        ("2020-01-01", "2020-01-05", ["2020-01-06", 7]),
    ],
)
def test_fold_mask_init_fail_mismatched_types(first_train_timestamp, last_train_timestamp, target_timestamps):
    with pytest.raises(ValueError, match="All timestamps should be one of two possible types: pd.Timestamp or int"):
        _ = FoldMask(
            first_train_timestamp=first_train_timestamp,
            last_train_timestamp=last_train_timestamp,
            target_timestamps=target_timestamps,
        )


@pytest.mark.parametrize(
    "ts_name, horizon, fold_mask",
    [
        (
            "example_tsds",
            1,
            FoldMask(first_train_timestamp=None, last_train_timestamp="2020-01-05", target_timestamps=["2020-01-06"]),
        ),
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2020-01-01", last_train_timestamp="2020-01-05", target_timestamps=["2020-01-06"]
            ),
        ),
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2020-01-03", last_train_timestamp="2020-01-05", target_timestamps=["2020-01-06"]
            ),
        ),
        (
            "example_tsds",
            10,
            FoldMask(first_train_timestamp=None, last_train_timestamp="2020-01-05", target_timestamps=["2020-01-10"]),
        ),
        (
            "example_tsds",
            10,
            FoldMask(
                first_train_timestamp=None,
                last_train_timestamp="2020-01-05",
                target_timestamps=["2020-01-10", "2020-01-12"],
            ),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=None, last_train_timestamp=15, target_timestamps=[16]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=10, last_train_timestamp=15, target_timestamps=[16]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=12, last_train_timestamp=15, target_timestamps=[16]),
        ),
        (
            "example_tsds_int_timestamp",
            10,
            FoldMask(first_train_timestamp=None, last_train_timestamp=15, target_timestamps=[20]),
        ),
        (
            "example_tsds_int_timestamp",
            10,
            FoldMask(first_train_timestamp=None, last_train_timestamp=15, target_timestamps=[20, 22]),
        ),
    ],
)
def test_fold_mask_validate_on_dataset_ok(ts_name, fold_mask, horizon, request):
    ts = request.getfixturevalue(ts_name)
    fold_mask.validate_on_dataset(ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, horizon, fold_mask",
    [
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2019-01-01", last_train_timestamp="2020-01-05", target_timestamps=["2020-01-06"]
            ),
        ),
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2021-01-01", last_train_timestamp="2021-01-05", target_timestamps=["2021-01-06"]
            ),
        ),
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2020-01-01 01:00",
                last_train_timestamp="2020-01-05",
                target_timestamps=["2020-01-06"],
            ),
        ),
        (
            "example_tsds",
            1,
            FoldMask(first_train_timestamp=5, last_train_timestamp=15, target_timestamps=[16]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=5, last_train_timestamp=15, target_timestamps=[16]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=210, last_train_timestamp=215, target_timestamps=[216]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(
                first_train_timestamp="2020-01-01", last_train_timestamp="2020-01-05", target_timestamps=["2020-01-06"]
            ),
        ),
    ],
)
def test_fold_mask_validate_on_dataset_fail_not_present_first_train_timestamp(ts_name, fold_mask, horizon, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="First train timestamp isn't present in a given dataset"):
        fold_mask.validate_on_dataset(ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, horizon, fold_mask",
    [
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2020-01-01", last_train_timestamp="2021-01-05", target_timestamps=["2021-01-06"]
            ),
        ),
        (
            "example_tsds",
            1,
            FoldMask(first_train_timestamp=None, last_train_timestamp="2021-01-05", target_timestamps=["2021-01-06"]),
        ),
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp=None,
                last_train_timestamp="2020-01-05 01:00",
                target_timestamps=["2020-01-06"],
            ),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=10, last_train_timestamp=215, target_timestamps=[216]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=None, last_train_timestamp=215, target_timestamps=[216]),
        ),
    ],
)
def test_fold_mask_validate_on_dataset_fail_not_present_last_train_timestamp(ts_name, fold_mask, horizon, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Last train timestamp isn't present in a given dataset"):
        fold_mask.validate_on_dataset(ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, horizon, fold_mask",
    [
        (
            "example_tsds",
            1,
            FoldMask(
                first_train_timestamp="2020-01-01", last_train_timestamp="2020-01-05", target_timestamps=["2021-01-06"]
            ),
        ),
        (
            "example_tsds",
            1,
            FoldMask(first_train_timestamp=None, last_train_timestamp="2020-01-05", target_timestamps=["2021-01-06"]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=10, last_train_timestamp=15, target_timestamps=[216]),
        ),
        (
            "example_tsds_int_timestamp",
            1,
            FoldMask(first_train_timestamp=None, last_train_timestamp=15, target_timestamps=[216]),
        ),
    ],
)
def test_fold_mask_validate_on_dataset_fail_not_present_some_target_timestamps(ts_name, fold_mask, horizon, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Some target timestamps aren't present in a given dataset"):
        fold_mask.validate_on_dataset(ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, horizon, fold_mask",
    [
        (
            "example_tsds",
            3,
            FoldMask(
                first_train_timestamp="2020-01-01", last_train_timestamp="2020-01-05", target_timestamps=["2020-01-10"]
            ),
        ),
        (
            "example_tsds",
            3,
            FoldMask(first_train_timestamp=None, last_train_timestamp="2020-01-05", target_timestamps=["2020-01-10"]),
        ),
        (
            "example_tsds",
            3,
            FoldMask(
                first_train_timestamp=None,
                last_train_timestamp="2020-01-05",
                target_timestamps=["2020-01-06", "2020-01-10"],
            ),
        ),
        (
            "example_tsds_int_timestamp",
            3,
            FoldMask(first_train_timestamp=10, last_train_timestamp=15, target_timestamps=[20]),
        ),
        (
            "example_tsds_int_timestamp",
            3,
            FoldMask(first_train_timestamp=None, last_train_timestamp=15, target_timestamps=[20]),
        ),
        (
            "example_tsds_int_timestamp",
            3,
            FoldMask(first_train_timestamp=None, last_train_timestamp=15, target_timestamps=[16, 20]),
        ),
    ],
)
def test_fold_mask_validate_on_dataset_fail_not_enough_horizon(ts_name, fold_mask, horizon, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Last target timestamp should be not later than"):
        fold_mask.validate_on_dataset(ts=ts, horizon=horizon)


@pytest.mark.parametrize(
    "ts_name, start_timestamp, end_timestamp, expected_start_timestamp, expected_end_timestamp",
    [
        ("example_tsds", None, None, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-09")),
        ("example_tsds", pd.Timestamp("2020-01-05"), None, pd.Timestamp("2020-01-05"), pd.Timestamp("2020-04-09")),
        ("example_tsds", "2020-01-05", None, pd.Timestamp("2020-01-05"), pd.Timestamp("2020-04-09")),
        ("example_tsds", None, pd.Timestamp("2020-04-05"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-05")),
        ("example_tsds", None, "2020-04-05", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-05")),
        (
            "example_tsds",
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-04-05"),
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-04-05"),
        ),
        ("example_tsds", "2020-01-05", "2020-04-05", pd.Timestamp("2020-01-05"), pd.Timestamp("2020-04-05")),
        (
            "example_tsds",
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-04-09"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-04-09"),
        ),
        ("ts_with_different_series_length", None, None, pd.Timestamp("2020-01-01 4:00"), pd.Timestamp("2020-02-01")),
        ("example_tsds_int_timestamp", None, None, 10, 109),
        ("example_tsds_int_timestamp", 15, None, 15, 109),
        ("example_tsds_int_timestamp", None, 100, 10, 100),
        ("example_tsds_int_timestamp", 15, 100, 15, 100),
    ],
)
def test_make_predict_timestamps_ok(
    ts_name, start_timestamp, end_timestamp, expected_start_timestamp, expected_end_timestamp, request
):
    ts = request.getfixturevalue(ts_name)

    start_timestamp, end_timestamp = BasePipeline._make_predict_timestamps(
        ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )

    assert start_timestamp == expected_start_timestamp
    assert end_timestamp == expected_end_timestamp


@pytest.mark.parametrize(
    "ts_name, start_timestamp",
    [("example_tsds", 10), ("example_tsds_int_timestamp", pd.Timestamp("2020-01-01"))],
)
def test_make_predict_timestamps_fail_wrong_start_timestamp_type(ts_name, start_timestamp, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Parameter start_timestamp has incorrect type"):
        _ = BasePipeline._make_predict_timestamps(ts=ts, start_timestamp=start_timestamp, end_timestamp=None)


@pytest.mark.parametrize(
    "ts_name, end_timestamp",
    [("example_tsds", 10), ("example_tsds_int_timestamp", pd.Timestamp("2020-01-01"))],
)
def test_make_predict_timestamps_fail_wrong_start_timestamp_type(ts_name, end_timestamp, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Parameter end_timestamp has incorrect type"):
        _ = BasePipeline._make_predict_timestamps(ts=ts, start_timestamp=None, end_timestamp=end_timestamp)


@pytest.mark.parametrize(
    "ts_name, start_timestamp",
    [("example_tsds", pd.Timestamp("2019-01-01")), ("example_tsds", "2019-01-01"), ("example_tsds_int_timestamp", 8)],
)
def test_make_predict_timestamps_fail_early_start(ts_name, start_timestamp, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Value of start_timestamp is less than beginning of some segments"):
        _ = BasePipeline._make_predict_timestamps(ts=ts, start_timestamp=start_timestamp, end_timestamp=None)


@pytest.mark.parametrize(
    "ts_name, end_timestamp",
    [("example_tsds", pd.Timestamp("2021-01-01")), ("example_tsds", "2021-01-01"), ("example_tsds_int_timestamp", 120)],
)
def test_make_predict_timestamps_fail_late_end(ts_name, end_timestamp, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Value of end_timestamp is more than ending of dataset"):
        _ = BasePipeline._make_predict_timestamps(ts=ts, start_timestamp=None, end_timestamp=end_timestamp)


@pytest.mark.parametrize(
    "ts_name, start_timestamp, end_timestamp",
    [
        ("example_tsds", pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-09")),
        ("example_tsds", "2020-01-10", "2020-01-09"),
        ("example_tsds_int_timestamp", 20, 19),
    ],
)
def test_make_predict_timestamps_fail_start_later_than_end(ts_name, start_timestamp, end_timestamp, request):
    ts = request.getfixturevalue(ts_name)
    with pytest.raises(ValueError, match="Value of end_timestamp is less than start_timestamp"):
        _ = BasePipeline._make_predict_timestamps(ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)


class DummyPipeline(BasePipeline):
    def fit(self, ts: TSDataset):
        self.ts = ts
        return self

    def _forecast(self, return_components: bool) -> TSDataset:
        return self.ts

    def _predict(self, return_components: bool) -> TSDataset:
        return self.ts

    def save(self, path: pathlib.Path):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: pathlib.Path) -> Any:
        raise NotImplementedError()

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        return {}


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (None, None),
        (pd.Timestamp("2020-01-02"), None),
        ("2020-01-02", None),
        (10, None),
        (None, pd.Timestamp("2020-02-01")),
        (None, "2020-02-01"),
        (None, 10),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-01")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-02-03")),
        ("2020-01-02", "2020-02-01"),
        (12, 100),
    ],
)
def test_predict_calls_make_timestamps(start_timestamp, end_timestamp, example_tsds):
    pipeline = DummyPipeline(horizon=1)

    pipeline._make_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(ts=example_tsds, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    pipeline._make_predict_timestamps.assert_called_once_with(
        ts=example_tsds, start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )


@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,)])
def test_predict_calls_validate_quantiles(quantiles, example_tsds):
    pipeline = DummyPipeline(horizon=1)

    pipeline._make_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(ts=example_tsds, quantiles=quantiles)

    pipeline._validate_quantiles.assert_called_once_with(quantiles=quantiles)


@pytest.mark.parametrize("prediction_interval", [False, True])
@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,)])
def test_predict_calls_private_predict(prediction_interval, quantiles, example_tsds):
    pipeline = DummyPipeline(horizon=1)

    start_timestamp = MagicMock()
    end_timestamp = MagicMock()
    pipeline._make_predict_timestamps = MagicMock(return_value=(start_timestamp, end_timestamp))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(ts=example_tsds, prediction_interval=prediction_interval, quantiles=quantiles)

    pipeline._predict.assert_called_once_with(
        ts=example_tsds,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
        return_components=False,
    )


@pytest.fixture
def ts_short_segment():
    df = pd.DataFrame(
        {
            "timestamp": list(pd.date_range(start="2000-01-01", periods=5, freq=pd.offsets.Day())) * 2,
            "segment": ["segment_1"] * 5 + ["short"] * 5,
            "target": [1] * 5 + [np.NAN, np.NAN, np.NAN, 1, 2],
        }
    )
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_empty_segment():
    df = pd.DataFrame(
        {
            "timestamp": list(pd.date_range(start="2000-01-01", periods=5, freq=pd.offsets.Day())) * 2,
            "segment": ["segment_1"] * 5 + ["empty"] * 5,
            "target": [1] * 5 + [np.NAN] * 5,
        }
    )
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=pd.offsets.Day())
    return ts


def test_validate_backtest_dataset_pass(ts_short_segment, n_folds=1, horizon=2, stride=1):
    BasePipeline._validate_backtest_dataset(ts_short_segment, n_folds=n_folds, horizon=horizon, stride=stride)


def test_validate_backtest_dataset_fails_short_segment(ts_short_segment, n_folds=1, horizon=3, stride=1):
    min_required_length = horizon + (n_folds - 1) * stride
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"All the series from feature dataframe should contain at least "
            f"{horizon} + {n_folds - 1} * {stride} = {min_required_length} timestamps; "
            f"series short does not."
        ),
    ):
        BasePipeline._validate_backtest_dataset(ts_short_segment, n_folds=n_folds, horizon=horizon, stride=stride)


def test_validate_backtest_dataset_fails_empty_segment(ts_empty_segment, n_folds=1, horizon=1, stride=1):
    min_required_length = horizon + (n_folds - 1) * stride
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"All the series from feature dataframe should contain at least "
            f"{horizon} + {n_folds - 1} * {stride} = {min_required_length} timestamps; "
            f"series empty does not."
        ),
    ):
        BasePipeline._validate_backtest_dataset(ts_empty_segment, n_folds=n_folds, horizon=horizon, stride=stride)
