from enum import Enum
from typing import List
from typing import Optional

import pandas as pd

from etna.datasets import TSDataset
from etna.transforms.base import ReversibleTransform


class BinaryOperator(str, Enum):
    """Enum for mathematical operators from pandas."""

    add = "+"
    sub = "-"
    mul = "*"
    div = "/"
    floordiv = "/="
    mod = "%"
    pow = "**"

    eq = "=="
    ne = "!="
    le = "<="
    lt = "<"
    ge = ">="
    gt = ">"

    def _missing_(cls, value):
        raise ValueError(f"Incorrect operand, literal {value} is unsupported")

    def perform(self, df: pd.DataFrame, left_operand: str, right_operand: str, out_column: str) -> pd.DataFrame:
        """Perform binary operation on passed dataframe."""
        pandas_operator = getattr(pd.DataFrame, self.name)
        df_left = df.loc[:, pd.IndexSlice[:, left_operand]].rename(columns={left_operand: out_column}, level="feature")
        df_right = df.loc[:, pd.IndexSlice[:, right_operand]].rename(
            columns={right_operand: out_column}, level="feature"
        )
        return pandas_operator(df_left, df_right)


class BinaryOperationTransform(ReversibleTransform):
    """Perform binary operation on the columns of dataset."""

    def __init__(self, left_column: str, right_column: str, operator: str, out_column: Optional[str] = None):
        """Create instance of BinaryOperationTransform.

        Parameters
        ----------
        left_column:
            Name of the left column
        right_column:
            Name of the right column
        operator:
            Operation to perform on the columns
        out_column:
            Resulting column name, if don't set, name will be `left_column operator right_column`.
            If out_column is left_column or right_column, apply changes to the existing column out_column, else create new column.
        """
        inverse_logic = {"+": "-", "-": "+", "*": "/", "/": "*"}
        super().__init__(required_features=[left_column, right_column])
        self.inplace_flag = (left_column == out_column) | (right_column == out_column)
        self.left_operand = left_column if (not self.inplace_flag or left_column != right_column) else right_column
        self.right_operand = right_column if (self.left_operand == left_column) else left_column
        if self.left_operand == self.right_operand:
            raise ValueError("You should use LambdaTransform, when you perform operation only with one column")
        self.operator = BinaryOperator(operator)
        self.out_column = (
            out_column if out_column is not None else self.left_operand + self.operator + self.right_operand
        )

        self._out_column_regressor: Optional[bool] = None
        self.inverse_operator = BinaryOperator(inverse_logic[operator]) if operator in inverse_logic.values() else None

    def fit(self, ts: TSDataset) -> "BinaryOperationTransform":
        """Fit the transform."""
        self._out_column_regressor = self.left_operand in ts.regressors and self.right_operand in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, df: pd.DataFrame) -> "BinaryOperationTransform":
        """Fit preprocess method, does nothing in ``BinaryOperationTransform`` case.

                Parameters
                ----------
                df:
                    dataframe with data.

                Returns
                -------
                result: ``BinaryOperationTransform``
                """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform operation on passed dataframe.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        : pd.Dataframe
            transformed dataframe
        """
        result = self.operator.perform(
            df=df,
            left_operand=self.left_operand,
            right_operand=self.right_operand,
            out_column=self.out_column,
        )
        if self.inplace_flag:
            df.loc[:, pd.IndexSlice[:, self.out_column]] = result
        else:
            df = pd.concat((df, result), axis=1)
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform reverse operation on passed dataframe.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        : pd.Dataframe
            transformed dataframe
        Raises
        ------
        ValueError:
            if out_column is not left_column or right_column
        ValueError:
            If initial operation is not '+', '-', '*' or '/'
        """
        if not self.inplace_flag or self.inverse_operator is None:
            raise ValueError(
                "We only support logic for inverse transform if out_column is left_column or right_column and it is '+', '-', '*', '/' operation"
            )
        df.loc[:, pd.IndexSlice[:, self.out_column]] = self.inverse_operator.perform(
            df=df, left_operand=self.out_column, right_operand=self.left_operand, out_column=self.out_column
        )
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self._out_column_regressor is None:
            raise ValueError("Transform is not fitted!")
        return [self.out_column] if self._out_column_regressor else []


all = ["BinaryOperationTransform"]
