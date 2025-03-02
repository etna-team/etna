import warnings
from abc import ABC
from typing import List
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna.transforms import IrreversibleTransform


class BaseFeatureSelectionTransform(IrreversibleTransform, ABC):
    """Base class for feature selection transforms."""

    def __init__(self, features_to_use: Union[List[str], Literal["all"]] = "all"):
        super().__init__(required_features="all")
        self.features_to_use = features_to_use
        self.selected_features: List[str] = []

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def _get_features_to_use(self, df: pd.DataFrame) -> List[str]:
        """Get list of features from the dataframe to perform the selection on."""
        features = set(df.columns.get_level_values("feature")) - {"target"}
        if self.features_to_use != "all":
            features = features.intersection(self.features_to_use)
            if sorted(features) != sorted(self.features_to_use):
                warnings.warn("Columns from feature_to_use which are out of dataframe columns will be dropped!")
        return sorted(features)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top_k features.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result: pd.DataFrame
            Dataframe with only selected features
        """
        rest_columns = set(df.columns.get_level_values("feature")) - set(self._get_features_to_use(df))
        selected_columns = sorted(self.selected_features + list(rest_columns))
        result = df.loc[:, pd.IndexSlice[:, selected_columns]]
        return result
