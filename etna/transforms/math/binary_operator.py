from enum import Enum
from typing import List
from typing import Optional

import pandas as pd

from etna.datasets import TSDataset
from etna.transforms.base import ReversibleTransform


class BinaryOperator(str, Enum):
    """Enum for mathematical operators from pandas."""

    #: Add operation, value: "+"
    add = "+"
    #: Subtraction operation, value: "-"
    sub = "-"
    #: Multiplication operation, value: "*"
    mul = "*"
    #: Division operation, value: "/"
    div = "/"
    #: Floordivision operation, value: "//"
    floordiv = "//"
    #: Module operation, value: "%"
    mod = "%"
    #: Pow operation, value: "**"
    pow = "**"
    #: Equal operation, value: "=="
    eq = "=="
    #: Not operation, value: "!="
    ne = "!="
    #: Less or equal operation, value: "<="
    le = "<="
    #: Less operation, value: "<"
    lt = "<"
    #: Greater or equal operation, value: ">="
    ge = ">="
    #: Greater operation, value: ">"
    gt = ">"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"Supported operands: {', '.join([repr(m.value) for m in cls])}.")

    def perform(self, df: pd.DataFrame, left_operand: str, right_operand: str, out_column: str) -> pd.DataFrame:
        """Perform binary operation on passed dataframe.

        - If during the operation a division by zero of a positive number occurs, writes +inf to this cell of the column, if negative - -inf, if 0/0 - nan.
        - In the case of raising a negative number to a non-integer power, writes nan to this cell of the column.

        Parameters
        ----------
        df:
            Source Dataframe
        left_operand:
            Name of the left column
        right_operand:
            Name of the right column
        out_column:
            Resulting column name, which contains the result of the operation operand(left, right)

        Returns
        -------
        :
            Column which contains result of operation
        """
        pandas_operator = getattr(pd.DataFrame, self.name)
        df_left = df.loc[:, pd.IndexSlice[:, left_operand]].rename(columns={left_operand: out_column}, level="feature")
        df_right = df.loc[:, pd.IndexSlice[:, right_operand]].rename(
            columns={right_operand: out_column}, level="feature"
        )
        return pandas_operator(df_left, df_right)


class BinaryOperationTransform(ReversibleTransform):
    """Perform binary operation on the columns of dataset.

    - Inverse_transform functionality is only supported for operations +, -, * , /.
    - If during the operation a division by zero of a positive number occurs, writes +inf to this cell of the column, if negative - -inf, if 0/0 - nan.
    - In the case of raising a negative number to a non-integer power, writes nan to this cell of the column.

    Different ``out_column`` dtype and result dtype in inplace operations
    could lead to unexpected behaviour in different ``pandas`` versions.

    Examples
    --------
    >>> import numpy as np
    >>> from etna.datasets import generate_ar_df, TSDataset
    >>> from etna.transforms import BinaryOperationTransform
    >>> df = generate_ar_df(start_time="2020-01-01", periods=30, freq="D", n_segments=1)
    >>> df["feature"] = np.full(30, 10)
    >>> df["target"] = np.full(30, 1)
    >>> ts = TSDataset(df, "D")
    >>> ts["2020-01-01":"2020-01-06", "segment_0", ["feature", "target"]]
    segment    segment_0
    feature      feature target
    timestamp
    2020-01-01        10    1.0
    2020-01-02        10    1.0
    2020-01-03        10    1.0
    2020-01-04        10    1.0
    2020-01-05        10    1.0
    2020-01-06        10    1.0
    >>> transformer = BinaryOperationTransform(left_column="feature", right_column="target", operator="+", out_column="target")
    >>> new_ts = transformer.fit_transform(ts=ts)
    >>> new_ts["2020-01-01":"2020-01-06", "segment_0", ["feature", "target"]]
    segment    segment_0
    feature      feature target
    timestamp
    2020-01-01        10   11.0
    2020-01-02        10   11.0
    2020-01-03        10   11.0
    2020-01-04        10   11.0
    2020-01-05        10   11.0
    2020-01-06        10   11.0
    """

    def __init__(self, left_column: str, right_column: str, operator: str, out_column: Optional[str] = None):
        """Create instance of BinaryOperationTransform.

        Parameters
        ----------
        left_column:
            Name of the left column
        right_column:
            Name of the right column
        operator:
            Operation to perform on the columns, see :py:class:`~etna.transforms.math.binary_operator.BinaryOperator`
        out_column:
            - Resulting column name, if don't set, name will be `left_column operator right_column`.
            - If out_column is left_column or right_column, apply changes to the existing column out_column, else create new column.
        """
        inverse_logic = {"+": "-", "-": "+", "*": "/", "/": "*"}
        super().__init__(required_features=[left_column, right_column])
        self._inplace_flag = (left_column == out_column) | (right_column == out_column)
        self.left_column = left_column
        self.right_column = right_column
        if self.left_column == self.right_column:
            raise ValueError("You should use LambdaTransform, when you perform operation only with one column")
        self.operator = BinaryOperator(operator)
        self.out_column = out_column if out_column is not None else self.left_column + self.operator + self.right_column

        self._in_column_regressor: Optional[bool] = None
        self.inverse_operator = BinaryOperator(inverse_logic[operator]) if operator in inverse_logic else None

    def fit(self, ts: TSDataset) -> "BinaryOperationTransform":
        """Fit the transform."""
        self._in_column_regressor = self.left_column in ts.regressors and self.right_column in ts.regressors
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
        :
            result
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
        :
            transformed dataframe
        """
        result = self.operator.perform(
            df=df,
            left_operand=self.left_column,
            right_operand=self.right_column,
            out_column=self.out_column,
        )
        if self._inplace_flag:
            df.loc[:, pd.IndexSlice[:, self.out_column]] = result
        else:
            df = pd.concat((df, result), axis=1)
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform reverse operation on passed dataframe.
        If

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
        if not self._inplace_flag:
            return df

        if self.inverse_operator is None:
            raise ValueError("We only support inverse transform if the original operation is .+, .-, .*, ./")

        support_column = self.left_column if (self.left_column != self.out_column) else self.right_column
        if self.operator in ["+", "*"]:
            df.loc[:, pd.IndexSlice[:, self.out_column]] = self.inverse_operator.perform(
                df=df, left_operand=self.out_column, right_operand=support_column, out_column=self.out_column
            )
        else:
            if self.right_column == self.out_column:
                if self.operator == "-":
                    df.loc[:, pd.IndexSlice[:, self.out_column]] = -df.loc[:, pd.IndexSlice[:, self.out_column]]
                else:
                    df.loc[:, pd.IndexSlice[:, self.out_column]] = 1 / df.loc[:, pd.IndexSlice[:, self.out_column]]
            df.loc[:, pd.IndexSlice[:, self.out_column]] = self.inverse_operator.perform(
                df=df, left_operand=self.out_column, right_operand=support_column, out_column=self.out_column
            )

        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self._in_column_regressor is None:
            raise ValueError("Transform is not fitted!")
        return [self.out_column] if self._in_column_regressor and not self._inplace_flag else []


all = ["BinaryOperationTransform"]
