import random
from pathlib import Path

import numpy as np
import pandas as pd
from transforms import LagTransform
from transforms import StandardScalerTransform

from etna.datasets import TSDataset
from etna.loggers import ClearMLLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.models.nn import MLPModel
from etna.pipeline import Pipeline

FILE_PATH = Path(__file__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def dataloader(file_path: Path, freq: str) -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


def backtest():
    set_seed()

    ts = dataloader(file_path=Path("../data/example_dataset.csv"), freq="D")

    pipeline = Pipeline(
        model=MLPModel(
            input_size=8,
            decoder_length=14,
            hidden_size=[7, 7],
            trainer_params=dict(max_epochs=30, enable_checkpointing=False),
            split_params=dict(train_size=0.75),
        ),
        transforms=[
            StandardScalerTransform(in_column="target"),
            LagTransform(in_column="target", lags=list(range(5, 13)), out_column="lag"),
        ],
        horizon=5,
    )

    cml_logger = ClearMLLogger(
        project_name="test/clearml_basic", tags=["test", "clearml", "concurrency"], config=pipeline.to_dict()
    )
    tslogger.add(cml_logger)

    pipeline.backtest(ts, n_jobs=5, metrics=[MAE()])


if __name__ == "__main__":
    backtest()
