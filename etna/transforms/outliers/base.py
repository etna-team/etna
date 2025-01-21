from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import check_new_segments


class OutliersTransform(ReversibleTransform, ABC):
    """Finds outliers in specific columns of DataFrame and replaces it with NaNs."""

    def __init__(self, in_column: str, ignore_flag_column: Optional[str] = None):
        """
        Create instance of OutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        ignore_flag_column:
            column name for skipping values from outlier check
        """
        required_features = [in_column]
        if ignore_flag_column:
            required_features.append(ignore_flag_column)

        super().__init__(required_features=required_features)
        self.in_column = in_column
        self.ignore_flag_column = ignore_flag_column

        self.segment_outliers: Optional[Dict[str, pd.Series]] = None

        self._fit_segments: Optional[List[str]] = None

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
        if self.ignore_flag_column is not None:
            if self.ignore_flag_column not in ts.features:
                raise ValueError(f'Name ignore_flag_column="{self.ignore_flag_column}" not find.')
            types_ignore_flag = ts[..., self.ignore_flag_column].isin([0, 1]).all(axis=0)
            if not all(types_ignore_flag):
                raise ValueError(
                    f'Columns ignore_flag contain non binary value: columns: "{self.ignore_flag_column}" in segment: {types_ignore_flag[~types_ignore_flag].index.get_level_values("segment").tolist()}'
                )

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
            if self.ignore_flag_column:
                available_points = set(df[df[segment, self.ignore_flag_column] == 0].index.values)
            else:
                available_points = index_set
            segment_outliers_timestamps = list(
                available_points.intersection(self.segment_outliers[segment].index.values)
            )

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
    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
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
