import reprlib
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_residuals(forecast_ts_list: List["TSDataset"], ts: "TSDataset") -> "TSDataset":
    """Get residuals for further analysis.

    Function keeps hierarchy, features in result dataset and removes target components.

    Parameters
    ----------
    forecast_ts_list:
        List of TSDataset with forecast for each fold from backtest
    ts:
        dataset of timeseries that has answers to forecast

    Returns
    -------
    :
        TSDataset with residuals in forecasts

    Raises
    ------
    KeyError:
        if segments of ``forecast_df`` and ``ts`` aren't the same
    """
    from etna.datasets import TSDataset

    # cast list of TSDataset to pd.DataFrame
    forecast_df = pd.concat([forecast.to_pandas() for forecast in forecast_ts_list], axis=0)

    # remove target components
    ts_copy = deepcopy(ts)
    ts_copy.drop_target_components()
    ts_copy.drop_prediction_intervals()

    # find the residuals
    true_df = ts_copy[forecast_df.index, :, :]
    if set(ts_copy.segments) != set(forecast_df.columns.get_level_values("segment").unique()):
        raise KeyError("Segments of `ts` and `forecast_df` should be the same")
    true_df.loc[:, pd.IndexSlice[ts.segments, "target"]] -= forecast_df.loc[:, pd.IndexSlice[ts.segments, "target"]]

    # make TSDataset
    new_ts = TSDataset(df=true_df, freq=ts_copy.freq, hierarchical_structure=ts_copy.hierarchical_structure)
    new_ts._known_future = deepcopy(ts_copy.known_future)
    new_ts._regressors = deepcopy(ts_copy.regressors)
    if ts._df_exog is not None:
        new_ts._df_exog = ts._df_exog.copy(deep=True)
    return new_ts


def _get_existing_intervals(ts: "TSDataset") -> Set[str]:
    """Get prediction intervals names that are present inside the TSDataset."""
    return set(ts.prediction_intervals_names)


def _select_prediction_intervals_names(
    forecast_results: Dict[str, "TSDataset"], quantiles: Optional[List[float]]
) -> List[str]:
    """Select prediction intervals names from the forecast results.

    Selected prediction intervals exist in each forecast.
    """
    intersection_intervals_set = set.intersection(
        *[_get_existing_intervals(forecast) for forecast in forecast_results.values()]
    )
    intersection_intervals = list(intersection_intervals_set)

    if quantiles is None:
        selected_intervals = intersection_intervals

    else:
        quantile_names = {f"target_{q:.4g}" for q in quantiles}
        selected_intervals = list(intersection_intervals_set.intersection(quantile_names))

        if len(selected_intervals) == 0:
            raise ValueError("Unable to find provided quantiles in the datasets!")

        non_existent = quantile_names - intersection_intervals_set
        if non_existent:
            warnings.warn(f"Quantiles {non_existent} do not exist in each forecast dataset. They will be dropped.")

    return selected_intervals


def _prepare_forecast_results(
    forecast_ts: Union["TSDataset", List["TSDataset"], Dict[str, "TSDataset"]]
) -> Dict[str, "TSDataset"]:
    """Prepare dictionary with forecasts results."""
    from etna.datasets import TSDataset

    if isinstance(forecast_ts, TSDataset):
        return {"1": forecast_ts}
    elif isinstance(forecast_ts, list) and len(forecast_ts) > 0:
        return {str(i + 1): forecast for i, forecast in enumerate(forecast_ts)}
    elif isinstance(forecast_ts, dict) and len(forecast_ts) > 0:
        return forecast_ts
    else:
        raise ValueError("Unknown type of `forecast_ts`")


def _validate_intersecting_folds(fold_numbers: pd.Series):
    """Validate if folds aren't intersecting."""
    fold_info = []
    for fold_number in fold_numbers.unique():
        fold_start = fold_numbers[fold_numbers == fold_number].index.min()
        fold_end = fold_numbers[fold_numbers == fold_number].index.max()
        fold_info.append({"fold_start": fold_start, "fold_end": fold_end})

    fold_info.sort(key=lambda x: x["fold_start"])

    for fold_info_1, fold_info_2 in zip(fold_info[:-1], fold_info[1:]):
        if fold_info_2["fold_start"] <= fold_info_1["fold_end"]:
            raise ValueError("Folds are intersecting")


def _check_metrics_df_empty_segments(metrics_df: pd.DataFrame, metric_name: str) -> None:
    """Check if there are segments without any non-missing metrics."""
    df = metrics_df[["segment", metric_name]]
    initial_segments = set(df["segment"].unique())
    df = df.dropna(subset=[metric_name])
    filtered_segments = set(df["segment"].unique())

    if initial_segments != filtered_segments:
        missing_segments = initial_segments - filtered_segments
        missing_segments_repr = reprlib.repr(missing_segments)
        warnings.warn(
            f"There are segments with all missing metric values, they won't be plotted: {missing_segments_repr}."
        )


def _check_metrics_df_same_folds_for_each_segment(metrics_df: pd.DataFrame, metric_name: str) -> None:
    """Check if the same set of folds is present for each segment."""
    if "fold_number" not in metrics_df.columns:
        return

    df = metrics_df[["segment", "fold_number", metric_name]]
    # we don't take into account segments without any non-missing metrics, they are handled by other check
    df = df.dropna(subset=[metric_name])
    num_unique = df.groupby("segment", group_keys=False)["fold_number"].apply(frozenset).nunique()
    if num_unique > 1:
        warnings.warn("Some segments have different set of folds to be aggregated on due to missing values.")
