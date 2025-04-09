from typing import TYPE_CHECKING
import pandas as pd
from etna.datasets import TSDataset


if TYPE_CHECKING:
    from etna.ensembles import EnsembleMixin


def check_backtest_return_type(backtest_result: dict, ensemble: "EnsembleMixin"):
    for key, value in backtest_result.items():
        match key:
            case "metrics" | "fold_info":
                assert isinstance(value, pd.DataFrame)
            case "pipelines":
                assert isinstance(value, list)
                for pipeline in value:
                    assert isinstance(pipeline, ensemble)
            case "forecasts":
                assert isinstance(value, list)
                for ts in value:
                    assert isinstance(ts, TSDataset)
