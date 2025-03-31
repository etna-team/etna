from enum import Enum
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from etna.datasets.utils import determine_freq
from etna.datasets.utils import determine_num_steps
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper

_DEFAULT_FREQ = object()


class DeseasonalModel(str, Enum):
    """Enum for different types of deseasonality model."""

    additive = "additive"
    multiplicative = "multiplicative"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types allowed."
        )


class _OneSegmentDeseasonalityTransform(OneSegmentTransform):
    def __init__(self, in_column: str, period: int, model: str = DeseasonalModel.additive):
        """
        Init _OneSegmentDeseasonalityTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            'additive' (default) or 'multiplicative'
        """
        self.in_column = in_column
        self.period = period
        self.model = DeseasonalModel(model)
        self._seasonal: Optional[pd.Series] = None
        self._freq_offset: Optional[pd.DateOffset] = _DEFAULT_FREQ  # type: ignore

    def _roll_seasonal(self, x: pd.Series) -> np.ndarray:
        """
        Roll out seasonal component by x's time index.

        Parameters
        ----------
        x:
            processed column

        Returns
        -------
        result:
            seasonal component
        """
        if self._seasonal is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling.")
        if self._seasonal.index[0] <= x.index[0]:
            shift = (
                -determine_num_steps(
                    start_timestamp=self._seasonal.index[0], end_timestamp=x.index[0], freq=self._freq_offset
                )
                % self.period
            )
        else:
            shift = (
                determine_num_steps(
                    start_timestamp=x.index[0], end_timestamp=self._seasonal.index[0], freq=self._freq_offset
                )
                % self.period
            )
        return np.resize(np.roll(self._seasonal, shift=shift), x.shape[0])

    def fit(self, df: pd.DataFrame) -> "_OneSegmentDeseasonalityTransform":
        """
        Perform seasonal decomposition.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result:
            instance after processing

        Raises
        ------
        ValueError:
            if input column contains NaNs in the middle of the series
        """
        self._freq_offset = determine_freq(df.index, freq_format="offset")

        df = df.loc[df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index()]
        if df[self.in_column].isnull().values.any():
            raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")
        self._seasonal = seasonal_decompose(
            x=df[self.in_column], model=self.model, period=self.period, filt=None, two_sided=False, extrapolate_trend=0
        ).seasonal[: self.period]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtract seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result:
            Dataframe with extracted features

        Raises
        ------
        ValueError:
            if input column contains zero or negative values
        """
        result = df
        seasonal = self._roll_seasonal(result[self.in_column])
        if self.model == "additive":
            result[self.in_column] -= seasonal
        else:
            if np.any(result[self.in_column] <= 0):
                raise ValueError(
                    "The input column contains zero or negative values,"
                    "but multiplicative seasonality can not work with such values."
                )
            result[self.in_column] /= seasonal
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result:
            Dataframe with extracted features

        Raises
        ------
        ValueError:
            if input column contains zero or negative values
        ValueError:
            if prediction intervals columns contains zero or negative values
        """
        result = df
        seasonal = self._roll_seasonal(result[self.in_column])

        for column_name in result.columns:
            if self.model == "additive":
                result.loc[:, column_name] += seasonal

            else:
                if np.any(result.loc[:, column_name] <= 0):
                    raise ValueError(
                        f"The `{column_name}` column contains zero or negative values,"
                        "but multiplicative seasonality can not work with such values."
                    )
                result.loc[:, column_name] *= seasonal

        return result


class DeseasonalityTransform(ReversiblePerSegmentWrapper):
    """Transform that uses :py:func:`statsmodels.tsa.seasonal.seasonal_decompose` to subtract seasonal component from the data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(self, in_column: str, period: int, model: Literal["additive", "multiplicative"] = "additive"):
        """
        Init DeseasonalityTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            'additive' (Y[t] = T[t] + S[t] + e[t], default option) or 'multiplicative' (Y[t] = T[t] * S[t] * e[t])
        """
        self.in_column = in_column
        self.period = period
        self.model = model
        super().__init__(
            transform=_OneSegmentDeseasonalityTransform(
                in_column=self.in_column,
                period=self.period,
                model=self.model,
            ),
            required_features=[self.in_column],
        )

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``model``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {"model": CategoricalDistribution(["additive", "multiplicative"])}
