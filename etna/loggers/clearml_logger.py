import base64
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union
from uuid import uuid4

import pandas as pd

from etna.loggers.base import BaseLogger

if TYPE_CHECKING:
    from clearml.task import Task

    from etna.datasets import TSDataset


class ClearMLLogger(BaseLogger):
    """ClearML logger.

    Note
    ----
    This logger requires ``clearml`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.

    Warning
    -------
    There is a possibility, that aggregated metrics charts may log incorrectly.
    For more details see `issue <https://github.com/etna-team/etna/issues/579>`_.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_name_prefix: str = "",
        task_type: str = "training",
        tags: Optional[Sequence[str]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        auto_connect_frameworks: Union[bool, Mapping[str, Union[bool, str, list]]] = False,
        auto_resource_monitoring: Union[bool, Mapping[str, Any]] = True,
        auto_connect_streams: Union[bool, Mapping[str, bool]] = True,
        plot: bool = True,
        table: bool = True,
        config: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
    ):
        """Create instance of ``ClearMLLogger``.

        Parameters
        ----------
        project_name:
            The name of the project in which the experiment will be created.
        task_name
            The name of Task (experiment).
        task_name_prefix:
            Prefix for the Task name field.
        task_type:
            The task type.
        tags:
             Add a list of tags (str) to the created Task.
        output_uri:
            The default location for output models and other artifacts.
        auto_connect_frameworks:
            Automatically connect frameworks.
        auto_resource_monitoring:
            Automatically create machine resource monitoring plots.
        auto_connect_streams:
            Control the automatic logging of stdout and stderr.
        plot:
            Indicator for making and sending plots.
        table:
            Indicator for making and sending tables.
        config:
            A dictionary-like object for saving inputs to your job,
            like hyperparameters for a model or settings for a data preprocessing job.
        save_dir:
            Path to the directory for saving intermediate data. Used only when logging DL models.
            Defaults to ``./tb_save``

        Notes
        -----
        For more details see `documentation <https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit>`_

        """
        super().__init__()
        self.project_name = project_name
        self.task_name = (
            task_name_prefix + base64.urlsafe_b64encode(uuid4().bytes).decode("utf8").rstrip("=\n")[:8]
            if task_name is None
            else task_name
        )
        self.task_name_prefix = task_name_prefix
        self.task_type = task_type
        self.tags = tags
        self.output_uri = output_uri
        self.auto_connect_frameworks = auto_connect_frameworks
        self.auto_resource_monitoring = auto_resource_monitoring
        self.auto_connect_streams = auto_connect_streams
        self.plot = plot
        self.table = table
        self.config = config
        self.save_dir = "./tb_save" if save_dir is None else save_dir

        self._pl_logger = None
        self._job_type: Optional[str] = None
        self._fold_id: Optional[Union[str, int]] = None

        self._task: Optional["Task"] = None
        self.init_task()

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """Log any event.

        This class logs string representation of a message.

        Parameters
        ----------
        msg:
            Message or dict to log
        kwargs:
            Additional parameters for particular implementation
        """
        level = kwargs.get("level", logging.INFO)
        print_console = kwargs.get("print_console", True)
        self._get_logger().report_text(
            msg=str(msg) if not isinstance(msg, str) else msg, level=level, print_console=print_console
        )

    def log_backtest_metrics(
        self, ts: "TSDataset", metrics_df: pd.DataFrame, forecast_ts_list: List["TSDataset"], fold_info_df: pd.DataFrame
    ):
        """Write metrics to logger.

        Parameters
        ----------
        ts:
            TSDataset to with backtest data
        metrics_df:
            Dataframe produced with :py:meth:`etna.pipeline.Pipeline._get_backtest_metrics`
        forecast_ts_list:
            List of TSDataset with forecast for each fold from backtest
        fold_info_df:
            Fold information from backtest
        """
        from etna.analysis import plot_backtest_interactive
        from etna.metrics.utils import aggregate_metrics_df

        if self._job_type is None:
            raise ValueError(
                "Experiment is not properly initialized! Ensure that ``start_experiment`` is called before logging backtest results."
            )

        logger = self._get_logger()

        if self.table:
            logger.report_table(title="Fold info", series=self._job_type, table_plot=fold_info_df)

        if self.plot:
            fig = plot_backtest_interactive(forecast_ts_list, ts, history_len=100)
            logger.report_plotly(title="Backtest forecast", series=self._job_type, figure=fig)

        metrics_dict = aggregate_metrics_df(metrics_df)
        for metric, value in metrics_dict.items():
            logger.report_single_value(name=metric, value=value)

    def log_backtest_run(self, metrics: pd.DataFrame, forecast: pd.DataFrame, test: pd.DataFrame):
        """Backtest metrics from one fold to logger.

        Parameters
        ----------
        metrics:
            Dataframe with metrics from backtest fold
        forecast:
            Dataframe with forecast
        test:
            Dataframe with ground truth
        """
        from etna.datasets import TSDataset
        from etna.metrics.utils import aggregate_metrics_df

        if self._job_type is None or self._fold_id is None:
            raise ValueError(
                "Experiment is not properly initialized! Ensure that ``start_experiment`` is called before logging backtest run."
            )

        columns_name = list(metrics.columns)
        metrics = metrics.reset_index()
        metrics.columns = ["segment"] + columns_name

        logger = self._get_logger()
        if self.table:
            logger.report_table(
                title="Metrics per fold", series=f"{self._job_type} Fold - {self._fold_id}", table_plot=metrics
            )
            logger.report_table(
                title="Forecasts per fold",
                series=f"{self._job_type} Fold - {self._fold_id}",
                table_plot=TSDataset.to_flatten(forecast),
            )
            logger.report_table(
                title="Test folds",
                series=f"{self._job_type} Fold - {self._fold_id}",
                table_plot=TSDataset.to_flatten(test),
            )

        if isinstance(self._fold_id, int):
            metrics_dict = aggregate_metrics_df(metrics)
            for metric, value in metrics_dict.items():
                logger.report_scalar(title=metric, series=self._job_type, iteration=self._fold_id, value=value)
        else:
            self.log(
                msg="Logging of scalars requires integer fold ids! Aggregated metrics logging would be skipped!",
                level=logging.WARNING,
            )

    def start_experiment(self, job_type: Optional[str] = None, group: Optional[str] = None, *args, **kwargs):
        """Start ``Task``.

        Complete logger initialization or reinitialize it before the next experiment with the same name.

        Parameters
        ----------
        job_type:
            Specify the type of task, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual tasks into a larger experiment.
        """
        self._job_type = job_type

        if group is not None:
            try:
                self._fold_id = int(group)
            except ValueError:
                self._fold_id = group

        # Overwrites current pl_logger if any
        self._pl_logger = None

        if self._task is None:
            self.init_task()

    def init_task(self):
        """Reinit ``Task``."""
        from clearml import Task
        from clearml import TaskTypes

        auto_connect_frameworks = {
            "tensorboard": True,  # for DL models, that use pytorch lightning
            "joblib": True,  # for backtest with multiple jobs
        }
        if isinstance(self.auto_connect_frameworks, Mapping):
            auto_connect_frameworks = {**auto_connect_frameworks, **self.auto_connect_frameworks}

        self._task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            task_type=TaskTypes[self.task_type],
            tags=self.tags,
            output_uri=self.output_uri,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_resource_monitoring=self.auto_resource_monitoring,
            auto_connect_streams=self.auto_connect_streams,
            reuse_last_task_id=False,
        )
        if self.config is not None:
            self._task.connect(mutable=self.config)

    def finish_experiment(self, *args, **kwargs):
        """Finish ``Task``."""
        # flush all data for additionally spawned tasks
        if (self._task is not None) and (not self._task.is_main_task()):
            self._task.flush(wait_for_uploads=True)

        self._job_type = None
        self._fold_id = None
        self._pl_logger = None

    def _get_logger(self):
        """Return internal task logger."""
        if self._task is None:
            raise ValueError("ClearML task is not initialized!")
        return self._task.current_task().get_logger()

    @property
    def pl_logger(self):
        """Pytorch lightning loggers."""
        if self._pl_logger is None:
            from lightning.pytorch.loggers import TensorBoardLogger

            prefix = "" if self._fold_id is None else f"Fold-{self._fold_id}"
            self._pl_logger = TensorBoardLogger(
                save_dir=self.save_dir, name=self.task_name, prefix=prefix, version=prefix
            )

        return self._pl_logger
