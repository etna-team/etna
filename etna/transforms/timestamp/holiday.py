from enum import Enum
from typing import List
from typing import Optional

import holidays
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

_DEFAULT_FREQ = object()


# TODO: it shouldn't be called on freq=None, we should discuss this
def bigger_than_day(freq: Optional[str]):
    """Compare frequency with day."""
    dt = "2000-01-01"
    dates_day = pd.date_range(start=dt, periods=2, freq="D")
    dates_freq = pd.date_range(start=dt, periods=2, freq=freq)
    return dates_freq[-1] > dates_day[-1]


# TODO: it shouldn't be called on freq=None, we should discuss this
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


# TODO: discuss conceptual problems with
class HolidayTransform(IrreversibleTransform):
    """
    HolidayTransform generates series that indicates holidays in given dataset.

    * In ``binary`` mode shows the presence of holiday in that day.
    * In ``category`` mode shows the name of the holiday with value "NO_HOLIDAY" reserved for days without holidays.
    * In ``days_count`` mode shows the frequency of holidays in a given period.

      * If the frequency is weekly, then we count the proportion of holidays in a week (Monday-Sunday) that contains this day.
      * If the frequency is monthly, then we count the proportion of holidays in a month that contains this day.
      * If the frequency is yearly, then we count the proportion of holidays in a year that contains this day.
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
            internationally recognised codes, designated to country for which we want to find the holidays.
        mode:
            `binary` to indicate holidays, `category` to specify which holiday do we have at each day,
            `days_count` to determine the proportion of holidays in a given period of time.
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
        self._freq: Optional[str] = _DEFAULT_FREQ  # type: ignore
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
        if self.in_column is None:
            self.in_column_regressor = True
        else:
            self.in_column_regressor = self.in_column in ts.regressors
        self._freq = ts.freq
        super().fit(ts)
        return self

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

    def _compute_feature(self, timestamps: pd.Series) -> pd.Series:
        if bigger_than_day(self._freq) and self._mode is not HolidayTransformMode.days_count:
            raise ValueError("For binary and category modes frequency of data should be no more than daily.")

        if self._mode is HolidayTransformMode.days_count:
            date_offset = pd.tseries.frequencies.to_offset(self._freq)
            values = []
            for dt in timestamps:
                if dt is pd.NaT:
                    values.append(pd.NA)
                else:
                    start_date, end_date = define_period(date_offset, pd.Timestamp(dt), self._freq)
                    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                    count_holidays = sum(1 for d in date_range if d in self.holidays)
                    holidays_freq = count_holidays / date_range.size
                    values.append(holidays_freq)
            result = pd.Series(values)
        elif self._mode is HolidayTransformMode.category:
            values = []
            for t in timestamps:
                if t is pd.NaT:
                    values.append(pd.NA)
                elif t in self.holidays:
                    values.append(self.holidays[t])
                else:
                    values.append(self._no_holiday_name)
            result = pd.Series(values)
        elif self._mode is HolidayTransformMode.binary:
            result = pd.Series([int(x in self.holidays) if x is not pd.NaT else pd.NA for x in timestamps])
        else:
            assert_never(self._mode)

        return result

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Parameters
        ----------
        df:
            value series with index column in timestamp format

        Returns
        -------
        :
            pd.DataFrame with added holidays

        Raises
        ------
        ValueError:
            if transform isn't fitted
        ValueError:
            if the frequency is greater than daily and this is a ``binary`` or ``categorical`` mode
        ValueError:
            if the frequency is not weekly, monthly, quarterly or yearly and this is ``days_count`` mode
        """
        if self._freq is _DEFAULT_FREQ:
            raise ValueError("Transform is not fitted")

        out_column = self._get_column_name()
        if self.in_column is None:
            if pd.api.types.is_integer_dtype(df.index.dtype):
                raise ValueError("Transform can't work with integer index, parameter in_column should be set!")

            feature = self._compute_feature(timestamps=df.index).values
            cols = df.columns.get_level_values("segment").unique()
            encoded_matrix = feature.reshape(-1, 1).repeat(len(cols), axis=1)
            wide_df = pd.DataFrame(
                encoded_matrix,
                columns=pd.MultiIndex.from_product([cols, [out_column]], names=("segment", "feature")),
                index=df.index,
            )
        else:
            features = TSDataset.to_flatten(df=df, features=[self.in_column])
            features[out_column] = self._compute_feature(timestamps=features[self.in_column])
            features.drop(columns=[self.in_column], inplace=True)
            wide_df = TSDataset.to_dataset(features)

        if self._mode is HolidayTransformMode.binary or self._mode is HolidayTransformMode.category:
            wide_df = wide_df.astype("category")

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
