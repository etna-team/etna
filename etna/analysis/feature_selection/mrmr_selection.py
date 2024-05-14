import math
import warnings
from enum import Enum
from typing import List

import numpy as np
import pandas as pd


class AggregationMode(str, Enum):
    """Enum for different aggregation modes."""

    #: Mean aggregation.
    mean = "mean"

    #: Maximum aggregation.
    max = "max"

    #: Minimum aggregation.
    min = "min"

    #: Median aggregation.
    median = "median"


AGGREGATION_FN = {
    AggregationMode.mean: np.mean,
    AggregationMode.max: np.max,
    AggregationMode.min: np.min,
    AggregationMode.median: np.median,
}


def is_relevant(relevance, feature, atol):
    """Return True if feature is relevant, else return False."""
    return not math.isclose(relevance.loc[feature], 0, abs_tol=atol)


def is_not_relevant(relevance, feature, atol):
    """Return False if feature is relevant, else return True."""
    return math.isclose(relevance.loc[feature], 0, abs_tol=atol)


def mrmr(
    relevance_table: pd.DataFrame,
    regressors: pd.DataFrame,
    top_k: int,
    fast_redundancy: bool = False,
    drop_zero: bool = True,
    relevance_aggregation_mode: str = AggregationMode.mean,
    redundancy_aggregation_mode: str = AggregationMode.mean,
    atol: float = 1e-10,
) -> List[str]:
    """
    Maximum Relevance and Minimum Redundancy feature selection method.

    Here relevance for each regressor is calculated as the per-segment aggregation of the relevance
    values in relevance_table. The redundancy term for the regressor is calculated as a mean absolute correlation
    between this regressor and other ones. The correlation between the two regressors is an aggregated pairwise
    correlation for the regressors values in each segment.

    Parameters
    ----------
    relevance_table:
        dataframe of shape n_segment x n_exog_series with relevance table, where ``relevance_table[i][j]``
        contains relevance of j-th ``df_exog`` series to i-th df series
    regressors:
        dataframe with regressors in etna format
    top_k:
        num of regressors to select; if there are not enough regressors, then all will be selected
    fast_redundancy:
        * True: compute redundancy only inside the the segments, time complexity :math:`O(top\_k * n\_segments * n\_features * history\_len)`
        * False: compute redundancy for all the pairs of segments, time complexity :math:`O(top\_k * n\_segments^2 * n\_features * history\_len)`
    drop_zero:
        If True, drop features with zero relevance before MRMR. If top_k is greater number of features
        with relevance > 0, select all this features and add features with zero relevance to select top_k features

    relevance_aggregation_mode:
        the method for relevance values per-segment aggregation
    redundancy_aggregation_mode:
        the method for redundancy values per-segment aggregation
    atol:
        the absolute tolerance to compare the float values

    Returns
    -------
    selected_features: List[str]
        list of ``top_k`` selected regressors, sorted by their importance
    """
    if not fast_redundancy:
        warnings.warn(
            "Option `fast_redundancy=False` was added for backward compatibility and will be removed in etna 3.0.0.",
            DeprecationWarning,
        )
    relevance_aggregation_fn = AGGREGATION_FN[AggregationMode(relevance_aggregation_mode)]
    redundancy_aggregation_fn = AGGREGATION_FN[AggregationMode(redundancy_aggregation_mode)]

    # can't compute correlation of categorical column with the others
    try:
        regressors = regressors.astype(float)
    except ValueError as e:
        raise ValueError(f"Only convertible to float features are allowed! Error: {str(e)}")

    relevance = relevance_table.apply(relevance_aggregation_fn).fillna(0)

    all_features = relevance.index.to_list()

    if top_k >= len(all_features):
        return all_features.copy()

    segments = set(regressors.columns.get_level_values("segment"))
    selected_features: List[str] = []
    not_selected_features = all_features.copy()

    if drop_zero is True:
        not_relevant_features = list(
            filter(lambda feature: is_not_relevant(relevance, feature, atol), not_selected_features)
        )
        not_selected_features = list(
            filter(lambda feature: is_relevant(relevance, feature, atol), not_selected_features)
        )
        if top_k >= len(not_selected_features):
            return not_selected_features + not_relevant_features[: (top_k - len(not_selected_features))]

    redundancy_table = pd.DataFrame(1, index=all_features, columns=all_features)

    for i in range(top_k):
        score_numerator = relevance.loc[not_selected_features]
        score_denominator = pd.Series(1, index=not_selected_features)
        if i > 0:
            last_selected_feature = selected_features[-1]
            last_selected_regressor = regressors.loc[pd.IndexSlice[:], pd.IndexSlice[:, last_selected_feature]]
            not_selected_regressors = regressors.loc[pd.IndexSlice[:], pd.IndexSlice[:, not_selected_features]]

            if fast_redundancy:
                segment_redundancy = pd.concat(
                    [
                        not_selected_regressors[segment].apply(
                            lambda col: last_selected_regressor[segment].corrwith(col)  # noqa: B023
                        )
                        for segment in segments
                    ]
                ).abs()
            else:
                segment_redundancy = (
                    not_selected_regressors.apply(lambda col: last_selected_regressor.corrwith(col))  # noqa: B023
                    .abs()
                    .groupby("feature")
                    .agg(redundancy_aggregation_fn)
                    .T.groupby("feature")
                )

            redundancy_table.loc[not_selected_features, last_selected_feature] = (
                segment_redundancy.agg(redundancy_aggregation_fn)
                .clip(atol)
                .fillna(1)
                .loc[not_selected_features]
                .values.squeeze()
            )

            score_denominator = redundancy_table.loc[not_selected_features, selected_features].max(axis=1)
        score = score_numerator * (1 - score_denominator)
        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    return selected_features
