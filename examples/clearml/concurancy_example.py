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
from joblib import Parallel,delayed

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




@hydra.main(config_name="config.yaml")
def backtest(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)

    logger = ClearMLLogger(project_name="a.p.chikov/test/clearml_basic", task_name="conc_example",
                           tags=["test", "clearml", "nn"], auto_connect_frameworks=True)
    def log(i, ob):
        logger.task.logger.report_scalar(title="lol", series=f"lol_{ob}", iteration=i, value=i)

    obj = np.random.randint(0, 3, size=1000)

    Parallel(n_jobs=3)(
        delayed(log)(i, ob) for i,ob in enumerate(obj)
    )




if __name__ == "__main__":
    backtest()
