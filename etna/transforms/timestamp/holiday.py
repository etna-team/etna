import datetime
from enum import Enum
from typing import List
from typing import Optional

import holidays
import pandas as pd

from etna.datasets import TSDataset
from etna.transforms.base import IrreversibleTransform


class HolidayTransformMode(str, Enum):
    """Enum for different imputation strategy."""

    binary = "binary"
    category = "category"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported mode: {', '.join([repr(m.value) for m in cls])}"
        )


class HolidayTransform(IrreversibleTransform):
    """
    HolidayTransform generates series that indicates holidays in given dataset.

    It doesn't work if distance between timestamps is higher than 1 day, for example week or month.

    In ``binary`` mode shows the presence of holiday in that day. In ``category`` mode shows the name of the holiday
    with value "NO_HOLIDAY" reserved for days without holidays.
    """

    _no_holiday_name: str = "NO_HOLIDAY"

    def __init__(
        self,
        iso_code: str = "RUS",
        mode: str = "binary",
        out_column: Optional[str] = None,
        in_column: Optional[str] = None,
    ):
        """
        Create instance of HolidayTransform.

        Parameters
        ----------
        iso_code:
            internationally recognised codes, designated to country for which we want to find the holidays
        mode:
            `binary` to indicate holidays, `category` to specify which holiday do we have at each day
        out_column:
            name of added column. Use ``self.__repr__()`` if not given.
        in_column:
            name of column to work with; if not given, index is used, only datetime index is supported
        """
        if in_column is None:
            required_features = ["target"]
        else:
            required_features = [in_column]
        super().__init__(required_features=required_features)

        self.iso_code = iso_code
        self.mode = mode
        self._mode = HolidayTransformMode(mode)
        self.holidays = holidays.country_holidays(iso_code)
        self.out_column = out_column
        self.in_column = in_column

        if self.in_column is None:
            self.in_column_regressor: Optional[bool] = True
        else:
            self.in_column_regressor = None

    def _get_column_name(self) -> str:
        if self.out_column:
            return self.out_column
        else:
            return self.__repr__()

    def fit(self, ts: TSDataset) -> "HolidayTransform":
        """Fit the transform."""
        if self.in_column is None:
            self.in_column_regressor = True
        else:
            self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, df: pd.DataFrame) -> "HolidayTransform":
        """
        Fit HolidayTransform with data from df. Does nothing in this case.

        Parameters
        ----------
        df:
            value series with index column in timestamp format
        """
        return self

    def _compute_feature(self, timestamp: pd.Series) -> pd.Series:
        if (timestamp[1] - timestamp[0]) > datetime.timedelta(days=1):
            raise ValueError("Frequency of data should be no more than daily.")

        if self._mode is HolidayTransformMode.category:
            values = []
            for t in timestamp:
                if t is pd.NaT:
                    values.append(pd.NA)
                elif t in self.holidays:
                    values.append(self.holidays[t])
                else:
                    values.append(self._no_holiday_name)
            result = pd.Series(values)
        else:
            result = pd.Series([int(x in self.holidays) if x is not pd.NaT else pd.NA for x in timestamp])

        return result

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data from df with HolidayTransform and generate a column of holidays flags or its titles.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format

        Returns
        -------
        :
            pd.DataFrame with added holidays
        """
        out_column = self._get_column_name()
        if self.in_column is None:
            if df.index.dtype == "int":
                raise ValueError("Transform can't work with integer index, parameter in_column should be set!")

            feature = self._compute_feature(timestamp=df.index).values
            cols = df.columns.get_level_values("segment").unique()
            encoded_matrix = feature.reshape(-1, 1).repeat(len(cols), axis=1)
            encoded_df = pd.DataFrame(
                encoded_matrix,
                columns=pd.MultiIndex.from_product([cols, [out_column]], names=("segment", "feature")),
                index=df.index,
            )
            encoded_df = encoded_df.astype("category")
            df = df.join(encoded_df).sort_index(axis=1)
        else:
            features = TSDataset.to_flatten(df=df, features=[self.in_column])
            features[out_column] = self._compute_feature(timestamp=features[self.in_column]).astype("category")
            features.drop(columns=[self.in_column], inplace=True)
            wide_df = TSDataset.to_dataset(features)
            df = pd.concat([df, wide_df], axis=1).sort_index(axis=1)
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.
        Returns
        -------
        :
            List with regressors created by the transform.
        """
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")

        if not self.in_column_regressor:
            return []

        return [self._get_column_name()]
