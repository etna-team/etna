import reprlib
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Set
from pandas._libs.tslibs.timestamps import Timestamp
import numpy as np
import pandas as pd
from bottleneck import nanmean

from etna.datasets import TSDataset
from etna.transforms import IrreversibleTransform


class ImputerMode(str, Enum):
    """Enum for different encoding strategies."""

    micro = "micro"
    macro = "macro"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"The strategy '{value}' doesn't exist")


class MissingMode(str, Enum):
    """Enum for handle missing strategies."""

    category = "category"
    global_mean = "global_mean"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported types: {', '.join([repr(m.value) for m in cls])}"
        )


class MeanEncoderTransform(IrreversibleTransform):
    """Makes expanding mean target encoding based on categorical column."""

    idx = pd.IndexSlice

    def __init__(
        self,
        in_column: str,
        out_column: str,
        mode: str = ImputerMode.micro,
        handle_missing: str = MissingMode.category,
        smoothing: int = 1,
    ):
        """
        Init MeanEncoderTransform.

        Parameters
        ----------
        in_column:
            categorical column to apply transform
        out_column:
            name of added column
        mode:
            mode to encode segments

            * 'micro' - running mean is calculated across each segment individually

            * 'macro' - running mean is calculated across all segments
        handle_missing:
            mode to handle missing values in ``in_column``

            * 'category' - NaNs they are interpreted as a separate categorical feature

            * 'global_mean' - NaNs are filled with global mean
        smoothing:
            smoothing parameter


        """
        super().__init__(required_features=["target", in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.mode = ImputerMode(mode)
        self.handle_missing = MissingMode(handle_missing)
        self.smoothing = smoothing

        self.global_means: Optional[Union[float, Dict[str, float]]] = None
        self.global_means_category: Optional[Union[Dict[str, float], Dict[str, Dict[str, float]]]] = None
        self.timestamps: Optional[Set[Timestamp]]

    def _fit(self, df: pd.DataFrame) -> "MeanEncoderTransform":
        """
        Fit encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit expanding mean target encoder.

        Returns
        -------
        :
            Fitted transform
        """
        self.timestamps = set(df.index)

        if self.mode is ImputerMode.micro:
            axis = 0
            segments = df.columns.get_level_values("segment").unique().tolist()
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)
            global_means = dict(zip(segments, global_means))

            global_means_category = {}
            for segment in segments:
                segment_df = TSDataset.to_flatten(df.loc[:, pd.IndexSlice[segment, :]])
                global_means_category[segment] = segment_df.groupby("regressor", dropna=False).mean().to_dict()["target"]
                if self.handle_missing is MissingMode.global_mean:
                    global_means_category[segment].discard(np.NaN)
        else:
            axis = None
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)

            segment_df = TSDataset.to_flatten(df)
            global_means_category = segment_df.groupby("regressor").mean().to_dict()["target"]
            if self.handle_missing is MissingMode.global_mean:
                global_means_category.discard(np.NaN)

        self.global_means = global_means
        self.global_means_category = global_means_category
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded values for the segment.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            result dataframe

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self.global_means is None:
            raise ValueError("The transform isn't fitted!")

        segments = df.columns.get_level_values("segment").unique().tolist()
        if self.mode is ImputerMode.micro:
            new_segments = set(segments) - self.global_means.keys()
            if len(new_segments) > 0:
                raise NotImplementedError(
                    f"This transform can't process segments that weren't present on train data: {reprlib.repr(new_segments)}"
                )

        flatten_df = TSDataset.to_flatten(df)
        flatten_df = flatten_df.sort_values(["timestamp", "segment"])
        flatten_df[self.out_column] = np.NaN
        if self.last_values is None:
            if self.mode is ImputerMode.micro:
                last_values = {}
                for segment in segments:
                    segment_df = flatten_df[flatten_df.segment == segment]
                    y = segment_df["target"]
                    temp = y.groupby(segment_df[self.in_column].astype(str)).agg(["cumsum", "cumcount"])
                    feature = (temp["cumsum"] - y + self.global_means[segment] * self.smoothing) / (
                        temp["cumcount"] + self.smoothing
                    )
                    flatten_df.loc[segment_df.index, self.out_column] = feature
                    if self.handle_missing is MissingMode.global_mean:
                        nan_index = segment_df[segment_df[self.in_column].isnull()].index
                        flatten_df.loc[nan_index, self.out_column] = self.global_means[segment]
                    segment_last_values = flatten_df.loc[segment_df.index].groupby(self.in_column, dropna=False).tail(1)
                    last_values[segment] = dict(
                        zip(segment_last_values[self.in_column], segment_last_values[self.out_column])
                    )
                self.last_values = last_values
            else:
                nan_index = flatten_df[flatten_df[self.in_column].isnull()].index
                y = flatten_df["target"]
                temp = y.groupby(flatten_df[self.in_column].astype(str)).agg(["cumsum", "cumcount"])
                feature = (temp["cumsum"] - y + self.global_means * self.smoothing) / (
                    temp["cumcount"] + self.smoothing
                )
                flatten_df[self.out_column] = feature
                if self.handle_missing is MissingMode.global_mean:
                    flatten_df.loc[nan_index, self.out_column] = self.global_means
                last_values = flatten_df.groupby(self.in_column, dropna=False).tail(1)
                self.last_values = dict(zip(last_values[self.in_column], last_values[self.out_column]))
        else:
            if self.mode is ImputerMode.micro:
                for segment in segments:
                    segment_df = flatten_df[flatten_df.segment == segment]
                    nan_index = segment_df[segment_df["target"].isnull()].index
                    flatten_df.loc[nan_index, self.out_column] = segment_df[self.in_column].map(
                        self.last_values[segment]
                    )
                    flatten_df.loc[nan_index, self.out_column] = flatten_df.loc[nan_index, self.out_column].fillna(
                        self.global_means[segment]
                    )
            else:
                nan_index = flatten_df[flatten_df["target"].isnull()].index
                flatten_df.loc[nan_index, self.out_column] = flatten_df.loc[nan_index, self.in_column].map(
                    self.last_values
                )
                flatten_df.loc[nan_index, self.out_column] = flatten_df.loc[nan_index, self.out_column].fillna(
                    self.global_means
                )
        df = TSDataset.to_dataset(flatten_df)
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self.out_column]
