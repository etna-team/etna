import warnings
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel


class SeasonalityMode(Enum):
    """Enum for seasonality mode for DeadlineMovingAverageModel."""

    month = "month"
    year = "year"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} seasonality allowed"
        )


class DeadlineMovingAverageModel(
    NonPredictionIntervalContextRequiredAbstractModel,
):
    """Moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
        """Initialize deadline moving average model.

        Length of the context is equal to the number of ``window`` months or years, depending on the ``seasonality``.

        Parameters
        ----------
        window:
            Number of values taken for forecast for each point.
        seasonality:
            Only allowed monthly or annual seasonality.
        """
        self.window = window
        self.seasonality = SeasonalityMode(seasonality)
        self._freqs_available = {"H", "D"}
        self._freq: Optional[str] = None

    def _validate_fitted(self):
        """Check if model is fitted."""
        if self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before trying the find out context size!")

    @property
    def context_size(self) -> int:
        """Upper bound to context size of the model."""
        self._validate_fitted()

        cur_value = None
        if self.seasonality is SeasonalityMode.year:
            cur_value = 366
        elif self.seasonality is SeasonalityMode.month:
            cur_value = 31

        if self._freq == "H":
            cur_value *= 24

        cur_value *= self.window

        return cur_value

    def get_model(self) -> "DeadlineMovingAverageModel":
        """Get internal model.

        Returns
        -------
        :
           Itself
        """
        return self

    def fit(self, ts: TSDataset) -> "DeadlineMovingAverageModel":
        """Fit model.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Model after fit
        """
        # we make a normalization to treat "1d" like "D"
        freq = pd.tseries.frequencies.to_offset(ts.freq).freqstr
        if freq not in self._freqs_available:
            raise ValueError(f"Freq {freq} is not supported! Use daily or hourly frequency!")

        self._freq = freq

        columns = set(ts.columns.get_level_values("feature"))
        if columns != {"target"}:
            warnings.warn(
                message=f"{type(self).__name__} does not work with any exogenous series or features. "
                f"It uses only target series for predict/\n "
            )
        return self

    @staticmethod
    def _get_context_beginning(
        df: pd.DataFrame, prediction_size: int, seasonality: SeasonalityMode, window: int
    ) -> pd.Timestamp:
        """Get timestamp where context begins.

        Parameters
        ----------
        df:
            Time series in a wide format.
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        seasonality:
            Seasonality.
        window:
            Number of values taken for forecast of each point.

        Returns
        -------
        :
            Timestamp with beginning of the context.

        Raises
        ------
        ValueError:
            if context isn't big enough
        """
        df_history = df.iloc[:-prediction_size]
        history_timestamps = df_history.index
        future_timestamps = df.iloc[-prediction_size:].index

        # if we have len(history_timestamps) == 0, then len(df) <= prediction_size
        if len(history_timestamps) == 0:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size or increase length of given dataframe!"
            )

        if seasonality is SeasonalityMode.month:
            first_index = future_timestamps[0] - pd.DateOffset(months=window)

        elif seasonality is SeasonalityMode.year:
            first_index = future_timestamps[0] - pd.DateOffset(years=window)

        if first_index < history_timestamps[0]:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size or increase length of given dataframe!"
            )

        return first_index

    def _make_predictions(
        self, result_template: pd.DataFrame, context: pd.DataFrame, prediction_size: int
    ) -> np.ndarray:
        """Make predictions using ``result_template`` as a base and ``context`` as a context."""
        index = result_template.index
        start_idx = len(result_template) - prediction_size
        end_idx = len(result_template)
        for i in range(start_idx, end_idx):
            for w in range(1, self.window + 1):
                if self.seasonality == SeasonalityMode.month:
                    prev_date = result_template.index[i] - pd.DateOffset(months=w)
                elif self.seasonality == SeasonalityMode.year:
                    prev_date = result_template.index[i] - pd.DateOffset(years=w)

                result_template.loc[index[i]] += context.loc[prev_date]

            result_template.loc[index[i]] = result_template.loc[index[i]] / self.window

        result_values = result_template.values[-prediction_size:]
        return result_values

    def _forecast(self, df: pd.DataFrame, prediction_size: int) -> pd.DataFrame:
        """Make autoregressive forecasts on a wide dataframe."""
        context_beginning = self._get_context_beginning(
            df=df, prediction_size=prediction_size, seasonality=self.seasonality, window=self.window
        )

        history = df.loc[:, pd.IndexSlice[:, "target"]]
        history = history.iloc[:-prediction_size]
        history = history.loc[history.index >= context_beginning]
        if np.any(history.isnull()):
            raise ValueError("There are NaNs in a forecast context, forecast method requires context to be filled!")

        num_segments = history.shape[1]
        index = pd.date_range(start=context_beginning, end=df.index[-1], freq=self._freq)
        result_template = np.append(history.values, np.zeros((prediction_size, num_segments)), axis=0)
        result_template = pd.DataFrame(result_template, index=index, columns=history.columns)
        result_values = self._make_predictions(
            result_template=result_template, context=result_template, prediction_size=prediction_size
        )

        df = df.iloc[-prediction_size:]
        y_pred = result_values[-prediction_size:]
        df.loc[:, pd.IndexSlice[:, "target"]] = y_pred
        return df

    def forecast(self, ts: TSDataset, prediction_size: int, return_components: bool = False) -> TSDataset:
        """Make autoregressive forecasts.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions

        Raises
        ------
        NotImplementedError:
            if return_components mode is used
        ValueError:
            if model isn't fitted
        ValueError:
            if context isn't big enough
        ValueError:
            if forecast context contains NaNs
        """
        if return_components:
            raise NotImplementedError("This mode isn't currently implemented!")
        self._validate_fitted()

        df = ts.to_pandas()
        new_df = self._forecast(df=df, prediction_size=prediction_size)
        ts.df = new_df
        return ts

    def _predict(self, df: pd.DataFrame, prediction_size: int) -> pd.DataFrame:
        """Make predictions on a wide dataframe using true values as autoregression context."""
        context_beginning = self._get_context_beginning(
            df=df, prediction_size=prediction_size, seasonality=self.seasonality, window=self.window
        )

        context = df.loc[:, pd.IndexSlice[:, "target"]]
        context = context.loc[context.index >= context_beginning]
        if np.any(context.isnull()):
            raise ValueError("There are NaNs in a target column, predict method requires target to be filled!")

        num_segments = context.shape[1]
        index = pd.date_range(start=df.index[-prediction_size], end=df.index[-1], freq=self._freq)
        result_template = pd.DataFrame(np.zeros((prediction_size, num_segments)), index=index, columns=context.columns)
        result_values = self._make_predictions(
            result_template=result_template, context=context, prediction_size=prediction_size
        )

        df = df.iloc[-prediction_size:]
        y_pred = result_values[-prediction_size:]
        df.loc[:, pd.IndexSlice[:, "target"]] = y_pred
        return df

    def predict(self, ts: TSDataset, prediction_size: int, return_components: bool = False) -> TSDataset:
        """Make predictions using true values as autoregression context (teacher forcing).

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.
        return_components:
            If True additionally returns prediction components

        Returns
        -------
        :
            Dataset with predictions

        Raises
        ------
        NotImplementedError:
            if return_components mode is used
        ValueError:
            if model isn't fitted
        ValueError:
            if context isn't big enough
        ValueError:
            if forecast context contains NaNs
        """
        if return_components:
            raise NotImplementedError("This mode isn't currently implemented!")
        self._validate_fitted()

        df = ts.to_pandas()
        new_df = self._predict(df=df, prediction_size=prediction_size)
        ts.df = new_df
        return ts


__all__ = ["DeadlineMovingAverageModel"]