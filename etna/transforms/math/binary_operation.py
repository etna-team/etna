from enum import Enum
from typing import List
from typing import Optional

import pandas as pd
from etna.datasets import TSDataset
from etna.transforms.base import ReversibleTransform


class MathOperator(str, Enum):
    """Enum for mathematical operators from pandas."""

    add = "+"
    sub = "-"
    mul = "*"
    div = "/"
    floordiv = "//"
    mod = "%"
    pow = "**"

    eq = "=="
    ne = "!="
    le = "<="
    lt = "<"
    ge = ">="
    gt = ">"

    def perform(self, df: pd.DataFrame, left_operand: str, right_operand: str, out_column: str) -> pd.DataFrame:
        """Perform mathematical operation on passed dataframe."""
        pandas_operator = getattr(pd.DataFrame, self.name)
        df_left = df.loc[:, pd.IndexSlice[:, left_operand]].rename(columns={left_operand: out_column}, level="feature")
        df_right = df.loc[:, pd.IndexSlice[:, right_operand]].rename(
            columns={right_operand: out_column}, level="feature"
        )
        return pandas_operator(df_left, df_right)


class BinaryOperationTransform(ReversibleTransform):
    """Perform binary mathematical operation on the columns of dataset."""

    def __init__(self, left_operand: str, right_operand: str, operator: str, out_column: Optional[str] = None):
        """Create instance of BinaryOperationTransform.

        Parameters
        ----------
        left_operand:
            Name of the left operand
        right_operand:
            Name of the right operand
        operator:
            Operation to perform on the operands
        out_column:
            Resulting column name, if don't set, name will be ``left_operand operator right_operand``
        """
        inverse_logic = {'+': '-', '-': '+', '*': '/', '/': "*"}
        super().__init__(required_features=[left_operand, right_operand])
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.operator = MathOperator(operator)
        self.out_column = (
            out_column if out_column is not None else self.left_operand + self.operator + self.right_operand
        )

        self._out_column_regressor: Optional[bool] = None
        self.inplace_flag = (left_operand == out_column) | (right_operand == out_column)
        inverse_operation = inverse_logic[operator] if operator in inverse_logic.values() else None
        self.inverse_operator = None
        if inverse_operation is not None:
            self.inverse_operator = MathOperator(inverse_operation)

    def fit(self, ts: TSDataset) -> "BinaryOperationTransform":
        """Fit the transform."""
        self._out_column_regressor = self.left_operand in ts.regressors and self.right_operand in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, df: pd.DataFrame):
        """Fits transform."""
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
            df=df, left_operand=self.left_operand, right_operand=self.right_operand, out_column=self.out_column,
        )
        if self.inplace_flag:
            df.loc[:, pd.IndexSlice[:, self.right_operand]] = result
        else:
            df = pd.concat((df, result), axis=1)
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace_flag or self.inverse_operator is None:
            raise ValueError("We only support logic for inverse transform if out_column is left_column or right_column and it is '+', '-', '*', '/' operation")
        df[self.out_column] = self.operator.perform(df =df, left_operand=self.right_operand, right_operand=self.left_operand, out_column=self.out_column)
        return df



    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self._out_column_regressor is None:
            raise ValueError("Transform is not fitted!")
        return [self.out_column] if self._out_column_regressor else []


__all__ = ["BinaryOperationTransform"]

