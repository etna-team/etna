import random
from typing import Optional

import hydra
import hydra_slayer
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pathlib import Path

from etna.datasets import TSDataset
from etna.loggers import ClearMLLogger
from etna.loggers import tslogger
from etna.pipeline import Pipeline
from etna.models.nn import MLPModel
from etna.transforms import LagTransform, StandardScalerTransform

FILE_PATH = Path(__file__)

OmegaConf.register_new_resolver("range", lambda x, y: list(range(x, y)))
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def init_logger(config: dict, project: str = "a.p.chikov/test/clearml_basic", tags: Optional[list] = ["test", "clearml", "nn"]):
    tslogger.loggers = []
    cml_logger = ClearMLLogger(project_name=project, tags=tags, config=config, auto_connect_frameworks=True)
    tslogger.add(cml_logger)


def dataloader(file_path: Path, freq: str) -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


@hydra.main(config_name="config.yaml")
def backtest(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Load data
    ts = dataloader(file_path=cfg.dataset.file_path, freq=cfg.dataset.freq)

    # Init pipeline
    pipeline: Pipeline = Pipeline(
        model=MLPModel(
            input_size=8,
            decoder_length=14,
            hidden_size=[7,7],
            train_batch_size=32,
            test_batch_size=64,
            trainer_params=dict(max_epochs=30),
        ),
        transforms= [
                StandardScalerTransform(in_column="target"),
                LagTransform(in_column="target", lags=list(range(14, 22)), out_column="lag")
            ],
        horizon=14,

    )

    ts.fit_transform(pipeline.transforms)
    model = pipeline.model
    
    # Init backtest parameters like metrics and e.t.c.
    backtest_params = hydra_slayer.get_from_params(**config["backtest"])

    # Init WandB logger
    init_logger(pipeline.to_dict())

    model.fit(ts)

    # Run backtest
    #_, _, _ = pipeline.backtest(ts, n_folds=1, **backtest_params)


if __name__ == "__main__":
    backtest()
