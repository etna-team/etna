from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd
from numpy.random import RandomState
from sklearn.ensemble import IsolationForest

from etna.datasets import TSDataset


def _select_features(
    ts: TSDataset, in_column: str, features_to_use: Optional[Sequence[str]], features_to_ignore: Optional[Sequence[str]]
) -> pd.DataFrame:
    features = ts.features
    if in_column not in features:
        raise ValueError(f"Feature {in_column} is not present in the dataset.")

    if features_to_use is None and features_to_ignore is None:
        return ts.to_pandas()

    df = ts.to_pandas()
    if features_to_use is not None and features_to_ignore is None:
        if not set(features_to_use).issubset(features):
            raise ValueError(f"Features {set(features_to_use) - set(features)} are not present in the dataset.")
        features_to_ignore = list(set(features) - set(features_to_use))
    elif features_to_ignore is not None and features_to_use is None:
        if not set(features_to_ignore).issubset(features):
            raise ValueError(f"Features {set(features_to_ignore) - set(features)} are not present in the dataset.")
    else:
        raise ValueError(
            "Changing the defaults there should be exactly one option set: features_to_use or features_to_ignore"
        )
    features_to_ignore = list(set(features_to_ignore) - {in_column})
    if len(features_to_ignore) > 0:
        df = df.drop(columns=features_to_ignore, level="feature")

    return df


def _prepare_segment_df(df: pd.DataFrame, segment: str, ignore_missing: bool) -> pd.DataFrame:
    df_segment = df[segment]
    if ignore_missing:
        return df_segment.dropna()

    first_valid_index = df_segment.isna().any(axis=1).idxmin()
    df_segment = df_segment.loc[first_valid_index:]
    if df_segment.isna().any().any():
        raise ValueError(
            f"Series {segment} contains NaNs! Set `ignore_missing=True` to drop them or impute them appropriately!"
        )
    return df_segment


def _get_anomalies_isolation_forest_segment(
    df_segment: pd.DataFrame, model: IsolationForest, in_column: str, use_in_column: bool, index_only: bool
) -> Union[List[pd.Timestamp], List[int], pd.Series]:
    model.fit(X=df_segment if use_in_column else df_segment.drop(columns=[in_column]))
    anomalies_flags = model.predict(X=df_segment if use_in_column else df_segment.drop(columns=[in_column])) == -1
    anomalies_series = df_segment.loc[anomalies_flags, in_column]
    if index_only:
        return list(anomalies_series.index.values)
    return anomalies_series


def get_anomalies_isolation_forest(
    ts: TSDataset,
    in_column: str = "target",
    features_to_use: Optional[Sequence[str]] = None,
    features_to_ignore: Optional[Sequence[str]] = None,
    ignore_missing: bool = False,
    n_estimators: int = 100,
    max_samples: Union[int, float, Literal["auto"]] = "auto",
    contamination: Union[float, Literal["auto"]] = "auto",
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[Union[int, RandomState]] = None,
    verbose: int = 0,
    index_only: bool = True,
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """
    Get point outliers in time series using Isolation Forest algorithm.

    `Documentation for Isolation Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`_.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        Name of the column in which the anomaly is searching
    features_to_use:
        List of feature column names to use for anomaly detection
    features_to_ignore:
        List of feature column names to exclude from anomaly detection
    ignore_missing:
        Whether to ignore missing values inside a series
    n_estimators:
        The number of base estimators in the ensemble
    max_samples:
        The number of samples to draw from X to train each base estimator
            *  If int, then draw max_samples samples.

            *  If float, then draw max_samples * X.shape[0] samples.

            *  If “auto”, then max_samples=min(256, n_samples).

        If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).
    contamination:
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
        Used when fitting to define the threshold on the scores of the samples.
            *  If ‘auto’, the threshold is determined as in the original paper.

            *  If float, the contamination should be in the range (0, 0.5].
    max_features:
        The number of features to draw from X to train each base estimator.
            *  If int, then draw max_features features.

            *  If float, then draw `max(1, int(max_features * n_features_in_))` features.
        Note: using a float number less than 1.0 or integer less than number of features
        will enable feature subsampling and leads to a longer runtime.
    bootstrap:
            *  If True, individual trees are fit on random subsets of the training data sampled with replacement.
            *  If False, sampling without replacement is performed.
    n_jobs:
        The number of jobs to run in parallel for both fit and predict.
            *  None means 1 unless in a joblib.parallel_backend context.
            *  -1 means using all processors
    random_state:
        Controls the pseudo-randomness of the selection of the feature and split values for
        each branching step and each tree in the forest.
    verbose:
        Controls the verbosity of the tree building process.
    index_only:
        whether to return only outliers indices. If `False` will return outliers series

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}
    """
    df = _select_features(
        ts=ts, in_column=in_column, features_to_use=features_to_use, features_to_ignore=features_to_ignore
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=False,
    )

    use_in_column = True
    if features_to_ignore is not None and in_column in features_to_ignore:
        use_in_column = False
    if features_to_use is not None and in_column not in features_to_use:
        use_in_column = False

    outliers_per_segment = {}
    for segment in ts.segments:
        df_segment = _prepare_segment_df(df=df, segment=segment, ignore_missing=ignore_missing)
        outliers_per_segment[segment] = _get_anomalies_isolation_forest_segment(
            df_segment=df_segment, model=model, in_column=in_column, use_in_column=use_in_column, index_only=index_only
        )

    return outliers_per_segment
