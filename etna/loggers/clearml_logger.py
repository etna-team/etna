import base64
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union
from uuid import uuid4

import pandas as pd

from etna import SETTINGS
from etna.loggers.base import BaseLogger

if TYPE_CHECKING:
    from etna.datasets import TSDataset

if SETTINGS.clearml_required:
    from clearml import Task
    from clearml import TaskTypes


class ClearMLLogger(BaseLogger):
    """ClearML logger.

    Note
    ----
    This logger requires ``clearml`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_name_prefix: str = "",
        task_type: Task.TaskTypes = TaskTypes.training,
        tags: Optional[Sequence[str]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        auto_connect_frameworks: Union[bool, Mapping[str, Union[bool, str, list]]] = False,
        auto_resource_monitoring: Union[bool, Mapping[str, Any]] = True,
        auto_connect_streams: Union[bool, Mapping[str, bool]] = True,
        plot: bool = True,
        table: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        # TODO: Is it enough?
        # TODO: Do we want same logic for PL, there is no ClearML Logger
        """
        Create instance of ClearMLLogger.

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

        Notes
        -----
        For more details see <https://clear.ml/docs/latest/docs/references/sdk/task#taskinit>

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

        self._task: Optional[Task] = None

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """
        Log any event.

        This class does nothing with it, use other loggers to do it.

        Parameters
        ----------
        msg:
            Message or dict to log
        kwargs:
            Additional parameters for particular implementation
        """
        pass

    def log_backtest_metrics(
        self, ts: "TSDataset", metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

        Parameters
        ----------
        ts:
            TSDataset to with backtest data
        metrics_df:
            Dataframe produced with :py:meth:`etna.pipeline.Pipeline._get_backtest_metrics`
        forecast_df:
            Forecast from backtest
        fold_info_df:
            Fold information from backtest
        """
        from etna.analysis import plot_backtest_interactive
        from etna.datasets import TSDataset
        from etna.metrics.utils import aggregate_metrics_df

        if self.table:
            self.task.logger.report_table(title="Metrics", series="Metrics", table_plot=metrics_df)
            self.task.logger.report_table(
                title="Forecast", series="Forecast", table_plot=TSDataset.to_flatten(forecast_df)
            )
            self.task.logger.report_table(title="Fold info", series="Fold info", table_plot=fold_info_df)

        if self.plot:
            fig = plot_backtest_interactive(forecast_df, ts, history_len=100)
            self.task.logger.report_plotly(title="Plot backtest forecast", series="Plot backtest forecast", figure=fig)

        metrics_dict = aggregate_metrics_df(metrics_df)
        for metric, value in metrics_dict.items():
            self.task.logger.report_single_value(name=metric, value=value)

    def log_backtest_run(self, metrics: pd.DataFrame, forecast: pd.DataFrame, test: pd.DataFrame):
        """
        Backtest metrics from one fold to logger.

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

        columns_name = list(metrics.columns)
        metrics = metrics.reset_index()
        metrics.columns = ["segment"] + columns_name
        if self.table:
            self.task.logger.report_table(title="Metrics", series="Metrics", table_plot=metrics)
            self.task.logger.report_table(
                title="Forecast", series="Forecast", table_plot=TSDataset.to_flatten(forecast)
            )
            self.task.logger.report_table(title="Test", series="Test", table_plot=TSDataset.to_flatten(test))

        metrics_dict = aggregate_metrics_df(metrics)
        for metric, value in metrics_dict.items():
            self.task.logger.report_single_value(name=metric, value=value)

    def start_task(self, task_type: Optional[str] = None, *args, **kwargs):
        """Start Task.

        Complete logger initialization or reinitialize it before the next experiment with the same name.

        Parameters
        ----------
        task_type:
            Specify the type of Task
        """
        self.task_type = task_type
        self.reinit_task()

    def reinit_task(self):
        """Reinit Task."""
        self._task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            task_type=self.task_type,
            tags=self.tags,
            output_uri=self.output_uri,
            auto_connect_frameworks=self.auto_connect_frameworks,
            auto_resource_monitoring=self.auto_resource_monitoring,
            auto_connect_streams=self.auto_connect_streams,
        )
        if self.config is not None:
            self._task.connect(mutable=self.config)

    def finish_task(self):
        """Finish Task."""
        self._task.close()

    @property
    def task(self):
        """Init Task."""
        if self._task is None:
            self.reinit_task()
        return self._task
