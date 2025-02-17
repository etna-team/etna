import warnings
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from typing_extensions import Literal

from etna.analysis import RelevanceTable
from etna.analysis.feature_selection.mrmr_selection import AggregationMode
from etna.analysis.feature_selection.mrmr_selection import mrmr
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.transforms.feature_selection import BaseFeatureSelectionTransform

TreeBasedRegressor = Union[
    DecisionTreeRegressor,
    ExtraTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    CatBoostRegressor,
]


class TreeFeatureSelectionTransform(BaseFeatureSelectionTransform):
    """Transform that selects features according to tree-based models feature importance.

    Notes
    -----
    Transform works with any type of features, however most of the models works only with regressors.
    Therefore, it is recommended to pass the regressors into the feature selection transforms.
    """

    def __init__(
        self,
        model: Union[Literal["catboost"], Literal["random_forest"], TreeBasedRegressor],
        top_k: int,
        features_to_use: Union[List[str], Literal["all"]] = "all",
    ):
        """
        Init TreeFeatureSelectionTransform.

        Parameters
        ----------
        model:
            Model to make selection, it should have ``feature_importances_`` property
            (e.g. all tree-based regressors in sklearn).

            If ``catboost.CatBoostRegressor`` is given with no ``cat_features`` parameter,
            then ``cat_features`` are set during ``fit`` to be equal to columns of category type.

            Pre-defined options are also available:

            * catboost: ``catboost.CatBoostRegressor(iterations=1000, silent=True)``;

            * random_forest: ``sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=0)``.

        top_k:
            num of features to select; if there are not enough features, then all will be selected
        features_to_use:
            columns of the dataset to select from; if "all" value is given, all columns are used
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Parameter top_k should be positive integer")
        super().__init__(features_to_use=features_to_use)
        self.top_k = top_k
        if isinstance(model, str):
            if model == "catboost":
                self.model = CatBoostRegressor(iterations=1000, silent=True)
            elif model == "random_forest":
                self.model = RandomForestRegressor(random_state=0)
            else:
                raise ValueError(f"Not a valid option for model: {model}")
        else:
            self.model = model

    def _get_train(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train data for model."""
        features = self._get_features_to_use(df)
        df = TSDataset.to_flatten(df).dropna()
        train_target = df["target"]
        train_data = df[features]
        return train_data, train_target

    def _get_features_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get weights for features based on model feature importances."""
        train_data, train_target = self._get_train(df)
        if isinstance(self.model, CatBoostRegressor) and self.model.get_param("cat_features") is None:
            dtypes = train_data.dtypes
            cat_features = dtypes[dtypes == "category"].index.tolist()
            self.model.fit(train_data, train_target, cat_features=cat_features)
        else:
            self.model.fit(train_data, train_target)
        weights_array = self.model.feature_importances_
        weights_dict = {column: weights_array[i] for i, column in enumerate(train_data.columns)}
        return weights_dict

    @staticmethod
    def _select_top_k_features(weights: Dict[str, float], top_k: int) -> List[str]:
        keys = np.array(list(weights.keys()))
        values = np.array(list(weights.values()))
        idx_sort = np.argsort(values)[::-1]
        idx_selected = idx_sort[:top_k]
        return keys[idx_selected].tolist()

    def _fit(self, df: pd.DataFrame) -> "TreeFeatureSelectionTransform":
        """
        Fit the model and remember features to select.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result:
            instance after fitting
        """
        if len(self._get_features_to_use(df)) == 0:
            warnings.warn("It is not possible to select features if there aren't any")
            return self
        weights = self._get_features_weights(df)
        self.selected_features = self._select_top_k_features(weights, self.top_k)
        return self

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``model``, ``top_k``. Other parameters are expected to be set by the user.

        For ``model`` parameter only pre-defined options are suggested.
        For ``top_k`` parameter the maximum suggested value is not greater than ``self.top_k``.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "model": CategoricalDistribution(["catboost", "random_forest"]),
            "top_k": IntDistribution(low=1, high=self.top_k),
        }


class MRMRFeatureSelectionTransform(BaseFeatureSelectionTransform):
    """Transform that selects features according to MRMR variable selection method adapted to the timeseries case.

    Notes
    -----
    Transform works with any type of features, however most of the models works only with regressors.
    Therefore, it is recommended to pass the regressors into the feature selection transforms.
    """

    def __init__(
        self,
        relevance_table: RelevanceTable,
        top_k: int,
        features_to_use: Union[List[str], Literal["all"]] = "all",
        fast_redundancy: bool = True,
        drop_zero: bool = False,
        relevance_aggregation_mode: str = AggregationMode.mean,
        redundancy_aggregation_mode: str = AggregationMode.mean,
        atol: float = 1e-10,
        **relevance_params,
    ):
        """
        Init MRMRFeatureSelectionTransform.

        Parameters
        ----------
        relevance_table:
            method to calculate relevance table
        top_k:
            num of features to select; if there are not enough features, then all will be selected
        features_to_use:
            columns of the dataset to select from
            if "all" value is given, all columns are used
        drop_zero:
            * True: use only features with relevance > 0 in calculations, if their number is less than ``top_k``,
              randomly selects features with zero relevance so that the total number of selected features is ``top_k``
            * False: use all features in calculations
        fast_redundancy:
            * True: compute redundancy only inside the segments, time complexity :math:`O(top\_k \\cdot n\_segments \\cdot n\_features \\cdot history\_len)`
            * False: compute redundancy for all the pairs of segments, time complexity :math:`O(top\_k \\cdot n\_segments^2 \\cdot n\_features \\cdot history\_len)`
        relevance_aggregation_mode:
            the method for relevance values per-segment aggregation
        redundancy_aggregation_mode:
            the method for redundancy values per-segment aggregation
        atol:
            the absolute tolerance to compare the float values
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Parameter top_k should be positive integer")

        super().__init__(features_to_use=features_to_use)
        self.relevance_table = relevance_table
        self.top_k = top_k
        self.fast_redundancy = fast_redundancy
        self.drop_zero = drop_zero
        self.relevance_aggregation_mode = relevance_aggregation_mode
        self.redundancy_aggregation_mode = redundancy_aggregation_mode
        self.atol = atol
        self.relevance_params = relevance_params

    def _fit(self, df: pd.DataFrame) -> "MRMRFeatureSelectionTransform":
        """
        Fit the method and remember features to select.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result:
            instance after fitting
        """
        features = self._get_features_to_use(df)
        df_target = df.loc[:, pd.IndexSlice[:, "target"]]
        df_target = df_target.loc[df_target.first_valid_index() :]
        df_features = df.loc[:, pd.IndexSlice[:, features]]
        df_features = df_features.loc[df_features.first_valid_index() :]
        relevance_table = self.relevance_table(df_target, df_features, **self.relevance_params)

        if not self.relevance_table.greater_is_better:
            min_relevance = relevance_table.values.min()
            max_relevance = relevance_table.values.max()
            relevance_table = max_relevance + min_relevance - relevance_table

        self.selected_features = mrmr(
            relevance_table=relevance_table,
            regressors=df_features,
            top_k=self.top_k,
            fast_redundancy=self.fast_redundancy,
            drop_zero=self.drop_zero,
            relevance_aggregation_mode=self.relevance_aggregation_mode,
            redundancy_aggregation_mode=self.redundancy_aggregation_mode,
            atol=self.atol,
        )
        return self

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes ``top_k`` parameter. Other parameters are expected to be set by the user.

        For ``top_k`` parameter the maximum suggested value is not greater than ``self.top_k``.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "top_k": IntDistribution(low=1, high=self.top_k),
            "relevance_aggregation_mode": CategoricalDistribution(["mean", "median", "max", "min"]),
            "redundancy_aggregation_mode": CategoricalDistribution(["mean", "median", "max", "min"]),
        }
