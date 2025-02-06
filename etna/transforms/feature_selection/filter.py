from typing import List
from typing import Optional
from typing import Sequence

import pandas as pd

from etna.transforms.base import IrreversibleTransform


class FilterFeaturesTransform(IrreversibleTransform):
    """Filters features in each segment of the dataframe."""

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ):
        """Create instance of FilterFeaturesTransform.

        Parameters
        ----------
        include:
            list of columns to pass through filter
        exclude:
            list of columns to not pass through
        Raises
        ------
        ValueError:
            if both options set or none of them
        """
        super().__init__(required_features="all")
        self.include: Optional[Sequence[str]] = None
        self.exclude: Optional[Sequence[str]] = None
        if include is not None and exclude is None:
            self.include = list(set(include))
        elif exclude is not None and include is None:
            self.exclude = list(set(exclude))
        else:
            raise ValueError("There should be exactly one option set: include or exclude")

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def _fit(self, df: pd.DataFrame) -> "FilterFeaturesTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: FilterFeaturesTransform
        """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter features according to include/exclude parameters.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        result = df
        features = df.columns.get_level_values("feature")
        if self.include is not None:
            if not set(self.include).issubset(features):
                raise ValueError(f"Features {set(self.include) - set(features)} are not present in the dataset.")
            result = result.loc[:, pd.IndexSlice[:, self.include]]
        if self.exclude is not None and self.exclude:
            if not set(self.exclude).issubset(features):
                raise ValueError(f"Features {set(self.exclude) - set(features)} are not present in the dataset.")
            result = df.drop(columns=self.exclude, level="feature")

        result = result.sort_index(axis=1)
        return result
