import random
from typing import Optional

# import hydra
# import hydra_slayer
import numpy as np
import pandas as pd
# from omegaconf import DictConfig
# from omegaconf import OmegaConf

from pathlib import Path

from etna.datasets import TSDataset
from etna.loggers import ClearMLLogger
from etna.loggers import tslogger
from etna.pipeline import Pipeline
from etna.models import NaiveModel
from etna.models.nn import MLPModel, DeepARModel
from etna.metrics import MAE
from transforms import LagTransform, StandardScalerTransform

# OmegaConf.register_new_resolver("range", lambda x, y: list(range(x, y)))
# OmegaConf.register_new_resolver("sum", lambda x, y: x + y)


FILE_PATH = Path(__file__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def init_logger(config: dict, project: str = "test/clearml_basic", tags: Optional[list] = ["test", "clearml", "concurrency"]):
    tslogger.loggers = []
    cml_logger = ClearMLLogger(
        project_name=project,
        tags=tags,
        config=config
    )
    tslogger.add(cml_logger)


def dataloader(file_path: Path, freq: str) -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


# @hydra.main(config_name="config.yaml")
def backtest():
    # Set seed for reproducibility
    set_seed()

    # Load data
    ts = dataloader(file_path=Path("../data/example_dataset.csv"), freq="D")

    # Init pipeline
    # pipeline: Pipeline = hydra_slayer.get_from_params(**config["pipeline"])
    pipeline = Pipeline(
        model=NaiveModel(),
        # model=MLPModel(
        #     input_size=8,
        #     decoder_length=14,
        #     hidden_size=[7, 7],
        #     trainer_params=dict(max_epochs=30),
        #     split_params=dict(train_size=0.75)
        # ),
        transforms=[
            StandardScalerTransform(in_column="target"),
            LagTransform(in_column="target", lags=list(range(5, 13)), out_column="lag")
        ],
        horizon=5
    )

    
    # Init backtest parameters like metrics and e.t.c.
    # backtest_params = hydra_slayer.get_from_params(**config["backtest"])
    backtest_params = {}

    # Init WandB logger
    init_logger(pipeline.to_dict())

    # pipeline.fit(ts)


    # Run backtest
    _, _, _ = pipeline.backtest(ts, n_jobs=3, metrics=[MAE()], joblib_params=dict(backend="loky"), **backtest_params)


if __name__ == "__main__":
    backtest()
