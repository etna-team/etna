from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn import preprocessing
from sklearn.utils._encode import _check_unknown
from sklearn.utils._encode import _encode
from typing_extensions import assert_never

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import IrreversibleTransform


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    new_value = "new_value"
    mean = "mean"
    none = "none"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"The strategy '{value}' doesn't exist")


class ReturnType(str, Enum):
    """Enum for data types of returned columns."""

    categorical = "categorical"
    numeric = "numeric"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported types: {', '.join([repr(m.value) for m in cls])}"
        )


class _LabelEncoder(preprocessing.LabelEncoder):
    def transform(self, y: pd.Series, strategy: ImputerMode):
        diff = _check_unknown(y, known_values=self.classes_)

        is_new_index = np.isin(y, diff)

        encoded = np.zeros(y.shape[0], dtype=float)
        encoded[~is_new_index] = _encode(y.iloc[~is_new_index], uniques=self.classes_, check_unknown=False).astype(
            float
        )

        if strategy is ImputerMode.none:
            filling_value = None
        elif strategy is ImputerMode.new_value:
            filling_value = -1
        elif strategy is ImputerMode.mean:
            filling_value = np.mean(encoded[~np.isin(y, diff)])
        else:
            assert_never(strategy)

        encoded[is_new_index] = filling_value
        return encoded


class LabelEncoderTransform(IrreversibleTransform):
    """Encode categorical feature with value between 0 and n_classes-1."""

    def __init__(
        self,
        in_column: str,
        out_column: Optional[str] = None,
        strategy: str = ImputerMode.mean,
        return_type: str = ReturnType.categorical,
    ):
        """
        Init LabelEncoderTransform.

        Parameters
        ----------
        in_column:
            Name of column to be transformed
        out_column:
            Name of added column. If not given, use ``self.__repr__()``
        strategy:
            Filling encoding in not fitted values:

            - If "new_value", then replace missing values with '-1'

            - If "mean", then replace missing values using the mean in encoded column

            - If "none", then replace missing values with None
        return_type:
            Data type of returned columns:

            - If "categorical", then returned columns will have "category" data type

            - If "numeric", then returned columns will have float data type

        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.strategy = ImputerMode(strategy)
        self.return_type = ReturnType(return_type)
        self.le = _LabelEncoder()
        self.in_column_regressor: Optional[bool] = None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        return [self._get_column_name()] if self.in_column_regressor else []

    def _fit(self, df: pd.DataFrame) -> "LabelEncoderTransform":
        """
        Fit Label encoder.

        Parameters
        ----------
        df:
            Dataframe with data to fit the transform
        Returns
        -------
        :
            Fitted transform
        """
        y = TSDataset.to_flatten(df)[self.in_column]
        self.le.fit(y=y)
        return self

    def fit(self, ts: TSDataset) -> "LabelEncoderTransform":
        """Fit the transform."""
        self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the ``in_column`` by fitted Label encoder.

        Parameters
        ----------
        df
            Dataframe with data to transform

        Returns
        -------
        :
            Dataframe with column with encoded values
        """
        out_column = self._get_column_name()
        result_df = TSDataset.to_flatten(df)
        result_df[out_column] = self.le.transform(result_df[self.in_column], self.strategy)

        if self.return_type is ReturnType.categorical:
            return_type = "category"
        elif self.return_type is ReturnType.numeric:
            return_type = "float"
        else:
            assert_never(self.return_type)

        result_df[out_column] = result_df[out_column].astype(return_type)
        result_df = TSDataset.to_dataset(result_df)
        return result_df

    def _get_column_name(self) -> str:
        """Get the ``out_column`` depending on the transform's parameters."""
        if self.out_column:
            return self.out_column
        return self.__repr__()

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes ``strategy`` parameter. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "strategy": CategoricalDistribution(["new_value", "mean"]),
        }


class OneHotEncoderTransform(IrreversibleTransform):
    """Encode categorical feature as a one-hot numeric features.

    If unknown category is encountered during transform, the resulting one-hot
    encoded columns for this feature will be all zeros.
    """

    def __init__(self, in_column: str, out_column: Optional[str] = None, return_type: str = ReturnType.categorical):
        """
        Init OneHotEncoderTransform.

        Parameters
        ----------
        in_column:
            Name of column to be encoded
        out_column:
            Prefix of names of added columns. If not given, use ``self.__repr__()``
        return_type:
            Data type of returned columns:

            - If "categorical", then returned columns will have "category" data type

            - If "numeric", then returned columns will have float data type
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.return_type = ReturnType(return_type)

        sklearn_version_tuple = tuple(map(int, sklearn_version.split(".")))
        encoder_params = {}
        if sklearn_version_tuple < (1, 2):
            encoder_params["sparse"] = False
        else:
            encoder_params["sparse_output"] = False
        self.ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", dtype=int, **encoder_params)

        self.in_column_regressor: Optional[bool] = None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        return self._get_out_column_names() if self.in_column_regressor else []

    def _fit(self, df: pd.DataFrame) -> "OneHotEncoderTransform":
        """
        Fit One Hot encoder.

        Parameters
        ----------
        df:
            Dataframe with data to fit the transform
        Returns
        -------
        :
            Fitted transform
        """
        x = TSDataset.to_flatten(df)[[self.in_column]]
        self.ohe.fit(X=x)
        return self

    def fit(self, ts: TSDataset) -> "OneHotEncoderTransform":
        """Fit the transform."""
        self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the `in_column` by fitted One Hot encoder.

        Parameters
        ----------
        df
            Dataframe with data to transform

        Returns
        -------
        :
            Dataframe with column with encoded values
        """
        result_df = TSDataset.to_flatten(df)
        x = result_df[[self.in_column]]
        out_columns = self._get_out_column_names()
        result_df[out_columns] = self.ohe.transform(X=x)

        if self.return_type == ReturnType.categorical:
            return_type = "category"
        elif self.return_type == ReturnType.numeric:
            return_type = "float"

        result_df[out_columns] = result_df[out_columns].astype(return_type)
        result_df = TSDataset.to_dataset(result_df)
        return result_df

    def _get_out_column_names(self) -> List[str]:
        """Get the list of ``out_column`` depending on the transform's parameters."""
        out_column = self.out_column if self.out_column is not None else self.__repr__()
        out_columns = [out_column + "_" + str(i) for i in range(len(self.ohe.categories_[0]))]
        return out_columns
