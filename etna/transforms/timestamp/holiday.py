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
from etna.datasets import duplicate_data
from etna.datasets.utils import determine_freq
from etna.transforms.base import IrreversibleTransform

_DEFAULT_FREQ = object()


def define_period(offset: pd.DateOffset, dt: pd.Timestamp):
    """Define start_date and end_date of period using dataset frequency."""
    offset_week = pd.offsets.Week(weekday=6)
    offset_year = pd.offsets.YearEnd()
    if isinstance(offset, Week) and offset.weekday == 6:
        start_date = dt - offset_week + pd.Timedelta(days=1)
        end_date = dt
    elif isinstance(offset, Week) and offset.weekday is not None:
        start_date = dt - offset_week + pd.Timedelta(days=1)
        end_date = dt + offset_week
    elif isinstance(offset, YearEnd) and offset.month == 12:
        start_date = dt - offset_year + pd.Timedelta(days=1)
        end_date = dt
    elif isinstance(offset, (YearBegin, YearEnd)):
        start_date = dt - offset_year + pd.Timedelta(days=1)
        end_date = dt + offset_year
    elif isinstance(offset, (MonthEnd, QuarterEnd, YearEnd)):
        start_date = dt - offset + pd.Timedelta(days=1)
        end_date = dt
    elif isinstance(offset, (MonthBegin, QuarterBegin, YearBegin)):
        start_date = dt
        end_date = dt + offset - pd.Timedelta(days=1)
    else:
        raise ValueError(
            f"Days_count mode works only with weekly, monthly, quarterly or yearly data. You have freq={offset.freqstr}"
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

    * In ``binary`` mode shows the presence of holiday in a given timestamp.
    * In ``category`` mode shows the name of the holiday in a given timestamp, the value "NO_HOLIDAY" is reserved for days without holidays.
    * In ``days_count`` mode shows the frequency of holidays in a given period.

      * If the frequency is weekly, then we count the proportion of holidays in a week (Monday-Sunday) that contains this day.
      * If the frequency is monthly, then we count the proportion of holidays in a month that contains this day.
      * If the frequency is yearly, then we count the proportion of holidays in a year that contains this day.

    Transform can accept timestamp data in two forms:

    - As index. In this case the dataset index is used to compute features.
      The features will be the same for each segment.

    - As external column. In this case for each segment its ``in_column`` will be used to compute features.
      In ``days_count`` mode it is expected that for all segments only one frequency is used.

    Notes
    -----
    During fitting int ``days_count`` mode the transform saves frequency. It is assumed to be the same during ``transform``.
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
        self._freq_offset: pd.DateOffset = _DEFAULT_FREQ  # type: ignore
        self.holidays = holidays.country_holidays(iso_code, language="en_US")
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

        Raises
        ------
        ValueError:
            if index timestamp is integer and ``in_column`` isn't set
        ValueError:
            if external timestamp isn't datetime
        ValueError
            if in ``days_count`` mode external timestamp doesn't have frequency
        ValueError
            if in ``days_count`` mode external timestamp doesn't have the same frequency for all segments
        """
        if self.in_column is None:
            if self._mode is HolidayTransformMode.days_count:
                if ts.freq is None:
                    raise ValueError("Transform can't work with integer index, parameter in_column should be set!")
                self._freq_offset = ts.freq_offset
            else:
                # set some value that doesn't really matter
                self._freq_offset = object()  # type: ignore
            self.in_column_regressor = True
        else:
            self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _validate_external_timestamps(self, df: pd.DataFrame):
        df = df.droplevel("feature", axis=1)

        # here we are assuming that every segment has the same timestamp dtype
        timestamp_dtype = df.dtypes.iloc[0]
        if not pd.api.types.is_datetime64_dtype(timestamp_dtype):
            raise ValueError("Transform can work only with datetime external timestamp!")

        if self._mode is HolidayTransformMode.binary or self._mode is HolidayTransformMode.category:
            return

        segments = df.columns.unique()
        freq_values = set()
        for segment in segments:
            timestamps = df[segment]
            timestamps = timestamps.loc[timestamps.first_valid_index() :]
            if len(timestamps) >= 3:
                cur_freq = pd.infer_freq(timestamps)
                if cur_freq is None:
                    raise ValueError(
                        f"Invalid in_column values! Datetime values should be regular timestamps with some frequency. "
                        f"This doesn't hold for segment {segment}"
                    )
                freq_values.add(cur_freq)

        if len(freq_values) > 1:
            raise ValueError(
                f"Invalid in_column values! Datetime values should have the same frequency for every segment. "
                f"Discovered frequencies: {freq_values}"
            )

    def _infer_external_freq(self, df: pd.DataFrame) -> pd.DateOffset:
        df = df.droplevel("feature", axis=1)
        # here we are assuming that every segment has the same timestamp freq
        sample_segment = df.columns[0]
        sample_timestamps = df[sample_segment]
        sample_timestamps = sample_timestamps.loc[sample_timestamps.first_valid_index() :]
        freq_offset = determine_freq(sample_timestamps, freq_format="offset")
        return freq_offset

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

        Raises
        ------
        ValueError:
            if external timestamp isn't datetime
        ValueError
            if in ``days_count`` mode external timestamp doesn't have frequency
        ValueError
            if in ``days_count`` mode external timestamp doesn't have the same frequency for all segments
        """
        if self.in_column is not None:
            self._validate_external_timestamps(df)
            if self._mode is HolidayTransformMode.days_count:
                self._freq_offset = self._infer_external_freq(df)
            else:
                # set some value that doesn't really matter
                self._freq_offset = object()  # type: ignore
        return self

    def _compute_feature(self, timestamps: pd.Series) -> pd.Series:
        dtype = "float"
        if self._mode is HolidayTransformMode.binary or self._mode is HolidayTransformMode.category:
            dtype = "category"

        if self._mode is HolidayTransformMode.days_count:
            values = []
            for dt in timestamps:
                if dt is pd.NaT:
                    values.append(np.NAN)
                else:
                    start_date, end_date = define_period(offset=self._freq_offset, dt=pd.Timestamp(dt))
                    date_range = pd.date_range(start=start_date, end=end_date, freq=pd.offsets.Day())
                    count_holidays = sum(1 for d in date_range if d in self.holidays)
                    holidays_freq = count_holidays / date_range.size
                    values.append(holidays_freq)
        elif self._mode is HolidayTransformMode.category:
            values = []
            for t in timestamps:
                if t is pd.NaT:
                    values.append(pd.NA)
                elif t in self.holidays:
                    values.append(self.holidays[t])
                else:
                    values.append(self._no_holiday_name)  # type: ignore
        elif self._mode is HolidayTransformMode.binary:
            values = [int(x in self.holidays) if x is not pd.NaT else pd.NA for x in timestamps]
        else:
            assert_never(self._mode)
        result = pd.Series(values, index=timestamps, dtype=dtype, name=self._get_column_name())

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
            if the frequency is not weekly, monthly, quarterly or yearly in ``days_count`` mode
        ValueError:
            if index timestamp is integer and ``in_column`` isn't set
        ValueError:
            if external timestamp isn't datetime
        ValueError
            if in ``days_count`` mode external timestamp doesn't have frequency
        ValueError
            if in ``days_count`` mode external timestamp doesn't have the same frequency for all segments
        """
        if self._freq_offset is _DEFAULT_FREQ:
            raise ValueError("Transform is not fitted")

        out_column = self._get_column_name()
        if self.in_column is None:
            if pd.api.types.is_integer_dtype(df.index.dtype):
                raise ValueError("Transform can't work with integer index, parameter in_column should be set!")

            feature = self._compute_feature(timestamps=df.index)
            segments = df.columns.get_level_values("segment").unique().tolist()
            wide_df = duplicate_data(df=feature.reset_index(), segments=segments)
        else:
            self._validate_external_timestamps(df=df)
            features = TSDataset.to_flatten(df=df, features=[self.in_column])
            features[out_column] = self._compute_feature(timestamps=features[self.in_column]).values
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
