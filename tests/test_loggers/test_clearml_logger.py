import os
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest
from pytorch_lightning.loggers import TensorBoardLogger

from etna.loggers import ClearMLLogger
from etna.loggers import tslogger as _tslogger
from etna.metrics import MAE
from etna.models import NaiveModel
from etna.models.nn import MLPModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform


@pytest.fixture()
def tslogger():
    _tslogger.loggers = []
    yield _tslogger
    _tslogger.loggers = []


@patch("etna.loggers.clearml_logger.ClearMLLogger.init_task")
def test_task_not_init_error(init_task):
    cml_logger = ClearMLLogger()
    with pytest.raises(ValueError, match="ClearML task is not initialized!"):
        cml_logger.log(msg="test")


@patch("etna.loggers.clearml_logger.ClearMLLogger.init_task")
def test_clearml_logger_log(init_task, tslogger):
    cml_logger = ClearMLLogger()
    cml_logger._task = MagicMock()

    idx = tslogger.add(cml_logger)
    tslogger.log("test")
    tslogger.log({"MAE": 0})
    tslogger.log({"MAPE": 1.5})
    tslogger.remove(idx)

    calls = [
        call(msg="test", level=20, print_console=True),
        call(msg="{'MAE': 0}", level=20, print_console=True),
        call(msg="{'MAPE': 1.5}", level=20, print_console=True),
    ]

    assert cml_logger._get_logger().report_text.call_count == 3
    cml_logger._get_logger().report_text.assert_has_calls(calls)


@pytest.mark.filterwarnings("ignore:The frame.append method is deprecated")
@patch("etna.loggers.clearml_logger.ClearMLLogger.init_task")
def test_default_pipeline(init_task, tslogger, example_tsds):
    cml_logger = ClearMLLogger()
    cml_logger._task = MagicMock()

    idx = tslogger.add(cml_logger)

    pipeline = Pipeline(model=NaiveModel(), transforms=[])
    pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=1)

    tslogger.remove(idx)

    # General checks
    assert cml_logger._get_logger().report_text.call_count > 0
    assert cml_logger._get_logger().report_table.call_count == 5 * 3 + 1  # n_folds + summary
    assert cml_logger._get_logger().report_plotly.call_count == 1  # single summary
    assert cml_logger._get_logger().report_single_value.call_count == 1 * 8  # final aggregated metric with stats
    assert cml_logger._get_logger().report_scalar.call_count == 5 * 8  # metric stats per each fold


@pytest.mark.filterwarnings("ignore:The frame.append method is deprecated")
@patch("etna.loggers.clearml_logger.ClearMLLogger.init_task")
def test_dl_pytorch_lightning_pipeline(init_task, tslogger, example_tsds):
    cml_logger = ClearMLLogger()
    cml_logger._task = MagicMock()

    idx = tslogger.add(cml_logger)

    pipeline = Pipeline(
        model=MLPModel(
            input_size=8,
            decoder_length=14,
            hidden_size=[7, 7],
            trainer_params=dict(max_epochs=3, enable_checkpointing=False, log_every_n_steps=1),
        ),
        transforms=[LagTransform(in_column="target", lags=list(range(5, 13)), out_column="lag")],
    )
    pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=1)

    tslogger.remove(idx)

    # General checks
    assert cml_logger._get_logger().report_text.call_count > 0
    assert cml_logger._get_logger().report_table.call_count == 5 * 3 + 1  # n_folds + summary
    assert cml_logger._get_logger().report_plotly.call_count == 1  # single summary
    assert cml_logger._get_logger().report_single_value.call_count == 1 * 8  # final aggregated metric with stats
    assert cml_logger._get_logger().report_scalar.call_count == 5 * 8  # metric stats per each fold

    # Lightning specific checks
    assert isinstance(cml_logger.pl_logger, TensorBoardLogger)
    assert os.path.exists(cml_logger.save_dir)
