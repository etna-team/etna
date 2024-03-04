from enum import Enum
from typing import List
from typing import Optional

import holidays
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.offsets import QuarterBegin
from pandas.tseries.offsets import QuarterEnd
from pandas.tseries.offsets import Week
from pandas.tseries.offsets import YearBegin
from pandas.tseries.offsets import YearEnd
from typing_extensions import assert_never

from etna.datasets import TSDataset
from etna.transforms.base import IrreversibleTransform


def bigger_than_day(freq: Optional[str]):
    """Compare frequency with day."""
    dt = "2000-01-01"
    dates_day = pd.date_range(start=dt, periods=2, freq="D")
    dates_freq = pd.date_range(start=dt, periods=2, freq=freq)
    return dates_freq[-1] > dates_day[-1]


def define_period(offset: pd.tseries.offsets.BaseOffset, dt: pd.Timestamp, freq: Optional[str]):
    """Define start_date and end_date of period using dataset frequency."""
    if isinstance(offset, Week) and offset.weekday == 6:
        start_date = dt - pd.tseries.frequencies.to_offset("W") + pd.Timedelta(days=1)
        end_date = dt
    elif isinstance(offset, Week):
        start_date = dt - pd.tseries.frequencies.to_offset("W") + pd.Timedelta(days=1)
        end_date = dt + pd.tseries.frequencies.to_offset("W")
    elif isinstance(offset, YearEnd) and offset.month == 12:
        start_date = dt - pd.tseries.frequencies.to_offset("Y") + pd.Timedelta(days=1)
        end_date = dt
    elif isinstance(offset, (YearBegin, YearEnd)):
        start_date = dt - pd.tseries.frequencies.to_offset("Y") + pd.Timedelta(days=1)
        end_date = dt + pd.tseries.frequencies.to_offset("Y")
    elif isinstance(offset, (MonthEnd, QuarterEnd, YearEnd)):
        start_date = dt - offset + pd.Timedelta(days=1)
        end_date = dt
    elif isinstance(offset, (MonthBegin, QuarterBegin, YearBegin)):
        start_date = dt
        end_date = dt + offset - pd.Timedelta(days=1)
    else:
        raise ValueError(
            f"Days_count mode works only with weekly, monthly, quarterly or yearly data. You have freq={freq}"
        )
    return start_date, end_date


class HolidayTransformMode(str, Enum):
    """Enum for different imputation strategy."""

    binary = "binary"
    category = "category"
    days_count = "days_count"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported mode: {', '.join([repr(m.value) for m in cls])}"
        )


class HolidayTransform(IrreversibleTransform):
    """
    HolidayTransform generates series that indicates holidays in given dataset.

    * In ``binary`` mode shows the presence of holiday in that day.
    * In ``category`` mode shows the name of the holiday with value "NO_HOLIDAY" reserved for days without holidays.
    * In ``days_count`` mode shows the frequency of holidays in a given period.
    * -
        - If the frequency is weekly, then we count the proportion of holidays in a week (Monday-Sunday) that contains this day.
        - If the frequency is monthly, then we count the proportion of holidays in a month that contains this day.
        - If the frequency is yearly, then we count the proportion of holidays in a year that contains this day.
    """

    _no_holiday_name: str = "NO_HOLIDAY"

    def __init__(self, iso_code: str = "RUS", mode: str = "binary", out_column: Optional[str] = None):
        """
        Create instance of HolidayTransform.

        Parameters
        ----------
        iso_code:
            internationally recognised codes, designated to country for which we want to find the holidays.
        mode:
            `binary` to indicate holidays, `category` to specify which holiday do we have at each day,
            `days_count` to determine the proportion of holidays in a given period of time.
        out_column:
            name of added column. Use ``self.__repr__()`` if not given.
        """
        super().__init__(required_features=["target"])
        self.iso_code = iso_code
        self.mode = mode
        self._mode = HolidayTransformMode(mode)
        self.holidays = holidays.country_holidays(iso_code)
        self.out_column = out_column
        self.freq: Optional[str] = None

    def _get_column_name(self) -> str:
        if self.out_column:
            return self.out_column
        else:
            return self.__repr__()

    def _fit(self, df: pd.DataFrame) -> "HolidayTransform":
        """Fit the transform.

        Parameters
        ----------
        df:
            Dataset to fit the transform on.

        Returns
        -------
        :
            The fitted transform instance.
        """
        return self

    def fit(self, ts: TSDataset):
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
        super().fit(ts=ts)
        self.freq = ts.freq
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format

        Returns
        -------
        :
            pd.DataFrame with added holidays

        Raises
        ------
        ValueError:
            if the frequency is greater than daily and this is a ``binary`` or ``categorical`` mode
        ValueError:
            if the frequency is not weekly, monthly, quarterly or yearly and this is ``days_count`` mode
        """
        if self.freq is None:
            raise ValueError("Transform is not fitted")
        if bigger_than_day(self.freq) and self._mode is not HolidayTransformMode.days_count:
            raise ValueError("For binary and category modes frequency of data should be no more than daily.")

        cols = df.columns.get_level_values("segment").unique()
        out_column = self._get_column_name()

        if self._mode is HolidayTransformMode.days_count:
            date_offset = pd.tseries.frequencies.to_offset(self.freq)
            encoded_matrix = np.empty(0)
            for dt in df.index:
                start_date, end_date = define_period(date_offset, pd.Timestamp(dt), self.freq)
                date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                count_holidays = sum(1 for d in date_range if d in self.holidays)
                holidays_freq = count_holidays / date_range.size
                encoded_matrix = np.append(encoded_matrix, holidays_freq)
        elif self._mode is HolidayTransformMode.category:
            encoded_matrix = np.array(
                [self.holidays[x] if x in self.holidays else self._no_holiday_name for x in df.index]
            )
        elif self._mode is HolidayTransformMode.binary:
            encoded_matrix = np.array([int(x in self.holidays) for x in df.index])
        else:
            assert_never(self._mode)
        encoded_matrix = encoded_matrix.reshape(-1, 1).repeat(len(cols), axis=1)
        encoded_df = pd.DataFrame(
            encoded_matrix,
            columns=pd.MultiIndex.from_product([cols, [out_column]], names=("segment", "feature")),
            index=df.index,
        )
        if self._mode is not HolidayTransformMode.days_count:
            encoded_df = encoded_df.astype("category")
        df = df.join(encoded_df)
        df = df.sort_index(axis=1)
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.
        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return [self._get_column_name()]
