from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from deprecated import deprecated

from etna.datasets import TSDataset
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import check_new_segments


class OutliersTransform(ReversibleTransform, ABC):
    """Finds outliers in specific columns of DataFrame and replaces it with NaNs."""

    def __init__(self, in_column: str):
        """
        Create instance of OutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column

        self.segment_outliers: Optional[Dict[str, pd.Series]] = None

        self._fit_segments: Optional[List[str]] = None

    @property
    @deprecated(
        reason="Attribute `outliers_timestamps` is deprecated and will be removed! Use `segment_outliers` instead.",
        version="3.0",
    )
    def outliers_timestamps(self) -> Union[Dict[str, List[pd.Timestamp]], Dict[str, List[int]], None]:
        """Backward compatibility property."""
        if self.segment_outliers is not None:
            return {segment: outliers.index.to_list() for segment, outliers in self.segment_outliers.items()}
        return None

    @property
    @deprecated(
        reason="Attribute `original_values` is deprecated and will be removed! Use `segment_outliers` instead.",
        version="3.0",
    )
    def original_values(self) -> Optional[Dict[str, pd.Series]]:
        """Backward compatibility property."""
        if self.segment_outliers is not None:
            return self.segment_outliers.copy()
        return None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.

        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return []

    def fit(self, ts: TSDataset) -> "OutliersTransform":
        """Fit the transform.

        Parameters
        ----------
        ts:
            Dataset to fit the transform on.

        Returns
        -------
        :
            The fitted transform instance.
        """
        self.segment_outliers = self.detect_outliers(ts)
        self._fit_segments = ts.segments
        super().fit(ts=ts)
        return self

    def _fit(self, df: pd.DataFrame) -> "OutliersTransform":
        """
        Find outliers using detection method.

        Parameters
        ----------
        df:
            dataframe with series to find outliers

        Returns
        -------
        result:
            instance with saved outliers
        """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace found outliers with NaNs.

        Parameters
        ----------
        df:
            transform ``in_column`` series of given dataframe

        Returns
        -------
        result:
            dataframe with in_column series with filled with NaNs

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self.segment_outliers is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")

        segments = set(df.columns.get_level_values("segment"))
        index_set = set(df.index.values)

        check_new_segments(transform_segments=segments, fit_segments=self._fit_segments)
        for segment in self.segment_outliers:
            if segment not in segments:
                continue
            # to locate only present indices
            segment_outliers_timestamps = list(index_set.intersection(self.segment_outliers[segment].index.values))
            df.loc[segment_outliers_timestamps, pd.IndexSlice[segment, self.in_column]] = np.NaN
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation. Returns back deleted values.

        Parameters
        ----------
        df:
            data to transform

        Returns
        -------
        result:
            data with reconstructed values

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self.segment_outliers is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling inverse_transform method.")

        segments = set(df.columns.get_level_values("segment"))
        index_set = set(df.index.values)

        check_new_segments(transform_segments=segments, fit_segments=self._fit_segments)
        for segment in self.segment_outliers:
            if segment not in segments:
                continue

            segment_outliers_timestamps = list(index_set.intersection(self.segment_outliers[segment].index.values))
            original_values = self.segment_outliers[segment][segment_outliers_timestamps].values
            df.loc[segment_outliers_timestamps, pd.IndexSlice[segment, self.in_column]] = original_values
        return df

    @abstractmethod
    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call function for detection outliers with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        pass
