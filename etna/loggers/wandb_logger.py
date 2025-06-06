import base64
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import uuid4

import pandas as pd

from etna import SETTINGS
from etna.loggers.base import BaseLogger

if TYPE_CHECKING:
    from lightning.pytorch.loggers import WandbLogger as PLWandbLogger

    from etna.datasets import TSDataset

if SETTINGS.wandb_required:
    import wandb


class WandbLogger(BaseLogger):
    """Weights&Biases logger.

    Note
    ----
    This logger requires ``wandb`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        job_type: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        plot: bool = True,
        table: bool = True,
        name_prefix: str = "",
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = False,
    ):
        """
        Create instance of WandbLogger.

        Parameters
        ----------
        name:
            Wandb run name.
        entity:
            An entity is a username or team name where you're sending runs.
        project:
            The name of the project where you're sending the new run
        job_type:
            Specify the type of run, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual runs into a larger experiment.
        tags:
            A list of strings, which will populate the list of tags on this run in the UI.
        plot:
            Indicator for making and sending plots.
        table:
            Indicator for making and sending tables.
        name_prefix:
            Prefix for the name field.
        config:
            This sets `wandb.config`, a dictionary-like object for saving inputs to your job,
            like hyperparameters for a model or settings for a data preprocessing job.
        log_model:
            Log checkpoints created by :py:class:`lightning.pytorch.callbacks.ModelCheckpoint`
            as W&B artifacts. `latest` and `best` aliases are automatically set.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              ``lightning.pytorch.callbacks.ModelCheckpoint.save_top_k==-1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.
        """
        super().__init__()
        self.name = (
            name_prefix + base64.urlsafe_b64encode(uuid4().bytes).decode("utf8").rstrip("=\n")[:8]
            if name is None
            else name
        )
        self.project = project
        self.entity = entity
        self.group = group
        self.config = config
        self._experiment = None
        self._pl_logger: Optional["PLWandbLogger"] = None
        self.job_type = job_type
        self.tags = tags
        self.plot = plot
        self.table = table
        self.name_prefix = name_prefix
        self.log_model = log_model

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """
        Log any event.

        e.g. "Fitted segment segment_name" to stderr output.

        Parameters
        ----------
        msg:
            Message or dict to log
        kwargs:
            Parameters for changing additional info in log message

        Notes
        -----
        We log dictionary to wandb only.
        """
        if isinstance(msg, dict):
            self.experiment.log(msg)

    def log_backtest_metrics(
        self, ts: "TSDataset", metrics_df: pd.DataFrame, forecast_ts_list: List["TSDataset"], fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

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

        summary: Dict[str, Any] = dict()
        if self.table:
            summary["metrics"] = wandb.Table(data=metrics_df)

            forecast_df = pd.concat(
                [
                    forecast_ts.to_pandas(flatten=True).assign(fold_number=num_fold)
                    for num_fold, forecast_ts in enumerate(forecast_ts_list)
                ],
                axis=0,
                ignore_index=True,
            )
            summary["forecast"] = wandb.Table(data=forecast_df)

            summary["fold_info"] = wandb.Table(data=fold_info_df)

        if self.plot:
            fig = plot_backtest_interactive(forecast_ts_list, ts, history_len=100)
            summary["backtest"] = fig

        metrics_dict = aggregate_metrics_df(metrics_df)
        summary.update(metrics_dict)
        self.experiment.log(summary)

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
        summary: Dict[str, Any] = dict()
        if self.table:
            summary["metrics"] = wandb.Table(data=metrics)
            summary["forecast"] = wandb.Table(data=TSDataset.to_flatten(forecast))
            summary["test"] = wandb.Table(data=TSDataset.to_flatten(test))

        metrics_dict = aggregate_metrics_df(metrics)
        for metric_key, metric_value in metrics_dict.items():
            summary[metric_key] = metric_value
        self.experiment.log(summary)

    def start_experiment(self, job_type: Optional[str] = None, group: Optional[str] = None, *args, **kwargs):
        """Start experiment.

        Complete logger initialization or reinitialize it before the next experiment with the same name.

        Parameters
        ----------
        job_type:
            Specify the type of run, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual runs into a larger experiment.
        """
        self.job_type = job_type
        self.group = group
        self.reinit_experiment()

    def reinit_experiment(self):
        """Reinit experiment."""
        self._experiment = wandb.init(
            name=self.name,
            project=self.project,
            entity=self.entity,
            group=self.group,
            config=self.config,
            reinit=True,
            tags=self.tags,
            job_type=self.job_type,
            settings=wandb.Settings(start_method="thread"),
        )

    def finish_experiment(self):
        """Finish experiment."""
        self._experiment.finish()

    @property
    def pl_logger(self):
        """Pytorch lightning loggers."""
        from lightning.pytorch.loggers import WandbLogger as PLWandbLogger

        self._pl_logger = PLWandbLogger(experiment=self.experiment, log_model=self.log_model)
        return self._pl_logger

    @property
    def experiment(self):
        """Init experiment."""
        if self._experiment is None:
            self.reinit_experiment()
        return self._experiment
