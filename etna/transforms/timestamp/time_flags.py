from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.datasets import duplicate_data
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import IrreversibleTransform


class TimeFlagsTransform(IrreversibleTransform):
    """TimeFlagsTransform is a class that implements extraction of the main time-based features from datetime column."""

    def __init__(
        self,
        minute_in_hour_number: bool = True,
        fifteen_minutes_in_hour_number: bool = False,
        hour_number: bool = True,
        half_hour_number: bool = False,
        half_day_number: bool = False,
        one_third_day_number: bool = False,
        out_column: Optional[str] = None,
        in_column: Optional[str] = None,
    ):
        """Initialise class attributes.

        Parameters
        ----------
        minute_in_hour_number:
            if True: add column with minute number to feature dataframe in transform
        fifteen_minutes_in_hour_number:
            if True: add column with number of fifteen-minute interval within hour with numeration from 0
            to feature dataframe in transform
        hour_number:
            if True: add column with hour number to feature dataframe in transform
        half_hour_number:
            if True: add column with 0 for the first half of the hour and 1 for the second
            to feature dataframe in transform
        half_day_number:
            if True: add column with 0 for the first half of the day and 1 for the second
            to feature dataframe in transform
        one_third_day_number:
            if True: add column with number of 8-hour interval within day with numeration from 0
            to feature dataframe in transform
        out_column:
            base for the name of created columns;

            * if set the final name is '{out_column}_{feature_name}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        in_column:
            name of column to work with; if not given, index is used, only datetime index is supported

        Raises
        ------
        ValueError:
            if all features aren't set in transform
        """
        if not any(
            [
                minute_in_hour_number,
                fifteen_minutes_in_hour_number,
                hour_number,
                half_hour_number,
                half_day_number,
                one_third_day_number,
            ]
        ):
            raise ValueError(
                f"{type(self).__name__} feature does nothing with given init args configuration, "
                f"at least one of minute_in_hour_number, fifteen_minutes_in_hour_number, hour_number, "
                f"half_hour_number, half_day_number, one_third_day_number should be True."
            )

        if in_column is None:
            required_features = ["target"]
        else:
            required_features = [in_column]
        super().__init__(required_features=required_features)

        self.date_column_name = None
        self.minute_in_hour_number: bool = minute_in_hour_number
        self.fifteen_minutes_in_hour_number: bool = fifteen_minutes_in_hour_number
        self.hour_number: bool = hour_number
        self.half_hour_number: bool = half_hour_number
        self.half_day_number: bool = half_day_number
        self.one_third_day_number: bool = one_third_day_number

        self.out_column = out_column
        self.in_column = in_column

        if self.in_column is None:
            self.in_column_regressor: Optional[bool] = True
        else:
            self.in_column_regressor = None

        # create empty init parameters
        self._empty_parameters = dict(
            minute_in_hour_number=False,
            fifteen_minutes_in_hour_number=False,
            hour_number=False,
            half_hour_number=False,
            half_day_number=False,
            one_third_day_number=False,
        )

    def _get_column_name(self, feature_name: str) -> str:
        if self.out_column is None:
            init_parameters = deepcopy(self._empty_parameters)
            init_parameters[feature_name] = getattr(self, feature_name)
            temp_transform = TimeFlagsTransform(**init_parameters, out_column=self.out_column)  # type: ignore
            return repr(temp_transform)
        else:
            return f"{self.out_column}_{feature_name}"

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")

        if not self.in_column_regressor:
            return []

        features = [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ]
        output_columns = [
            self._get_column_name(feature_name=feature_name) for feature_name in features if getattr(self, feature_name)
        ]
        return output_columns

    def fit(self, ts: TSDataset) -> "TimeFlagsTransform":
        """Fit the transform."""
        if self.in_column is None:
            self.in_column_regressor = True
        else:
            self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, *args, **kwargs) -> "TimeFlagsTransform":
        """Fit datetime model."""
        return self

    def _compute_features(self, timestamps: pd.Series) -> pd.DataFrame:
        timestamps_no_nans = timestamps.dropna()
        features = pd.DataFrame(index=timestamps_no_nans.index)

        if self.minute_in_hour_number:
            minute_in_hour_number = self._get_minute_number(timestamp_series=timestamps_no_nans)
            features[self._get_column_name("minute_in_hour_number")] = minute_in_hour_number

        if self.fifteen_minutes_in_hour_number:
            fifteen_minutes_in_hour_number = self._get_period_in_hour(
                timestamp_series=timestamps_no_nans, period_in_minutes=15
            )
            features[self._get_column_name("fifteen_minutes_in_hour_number")] = fifteen_minutes_in_hour_number

        if self.hour_number:
            hour_number = self._get_hour_number(timestamp_series=timestamps_no_nans)
            features[self._get_column_name("hour_number")] = hour_number

        if self.half_hour_number:
            half_hour_number = self._get_period_in_hour(timestamp_series=timestamps_no_nans, period_in_minutes=30)
            features[self._get_column_name("half_hour_number")] = half_hour_number

        if self.half_day_number:
            half_day_number = self._get_period_in_day(timestamp_series=timestamps_no_nans, period_in_hours=12)
            features[self._get_column_name("half_day_number")] = half_day_number

        if self.one_third_day_number:
            one_third_day_number = self._get_period_in_day(timestamp_series=timestamps_no_nans, period_in_hours=8)
            features[self._get_column_name("one_third_day_number")] = one_third_day_number

        for feature in features.columns:
            features[feature] = features[feature].astype("category")

        # add NaNs in features
        features = features.reindex(timestamps.index)

        return features

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method for features based on time.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        :
            Dataframe with extracted features
        """
        if self.in_column is None:
            if pd.api.types.is_integer_dtype(df.index.dtype):
                raise ValueError("Transform can't work with integer index, parameter in_column should be set!")

            timestamps = pd.Series(df.index)
            features = self._compute_features(timestamps=timestamps)
            features.index = df.index

            segments = df.columns.get_level_values("segment").unique().tolist()
            result = duplicate_data(df=features.reset_index(), segments=segments)
            result = pd.concat([df, result], axis=1).sort_index(axis=1)

        else:
            flat_df = TSDataset.to_flatten(df=df, features=[self.in_column])
            features = self._compute_features(timestamps=flat_df[self.in_column])
            features["timestamp"] = flat_df["timestamp"]
            features["segment"] = flat_df["segment"]
            wide_df = TSDataset.to_dataset(features)
            result = pd.concat([df, wide_df], axis=1).sort_index(axis=1)

        return result

    @staticmethod
    def _get_minute_number(timestamp_series: pd.Series) -> np.ndarray:
        """Generate array with the minute number in the hour."""
        return timestamp_series.apply(lambda x: x.minute).values

    @staticmethod
    def _get_period_in_hour(timestamp_series: pd.Series, period_in_minutes: int = 15) -> np.ndarray:
        """Generate an array with the period number in the hour.

        Accepts a period length in minutes as input and returns array where timestamps marked by period number.
        """
        return timestamp_series.apply(lambda x: x.minute // period_in_minutes).values

    @staticmethod
    def _get_hour_number(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the hour number in the day."""
        return timestamp_series.apply(lambda x: x.hour).values

    @staticmethod
    def _get_period_in_day(timestamp_series: pd.Series, period_in_hours: int = 12) -> np.ndarray:
        """Generate an array with the period number in the day.

        Accepts a period length in hours as input and returns array where timestamps marked by period number.
        """
        return timestamp_series.apply(lambda x: x.hour // period_in_hours).values

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``minute_in_hour_number``, ``fifteen_minutes_in_hour_number``, ``hour_number``,
        ``half_hour_number``, ``half_day_number``, ``one_third_day_number``.
        Other parameters are expected to be set by the user.

        There are no restrictions on all ``False`` values for the flags.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "minute_in_hour_number": CategoricalDistribution([False, True]),
            "fifteen_minutes_in_hour_number": CategoricalDistribution([False, True]),
            "hour_number": CategoricalDistribution([False, True]),
            "half_hour_number": CategoricalDistribution([False, True]),
            "half_day_number": CategoricalDistribution([False, True]),
            "one_third_day_number": CategoricalDistribution([False, True]),
        }


__all__ = ["TimeFlagsTransform"]
