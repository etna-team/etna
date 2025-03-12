from typing import List

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.datasets.utils import determine_num_steps
from etna.transforms import IrreversibleTransform


class FourierDecomposeTransform(IrreversibleTransform):
    """Transform that uses Fourier transformation to estimate series decomposition.

    Note
    ----
    This transform decomposes only in-sample data. For the future timestamps it produces ``NaN``.
    For the dataset to be transformed, it should contain at least the minimum amount of in-sample timestamps that are required by transform.

    Warning
    -------
    This transform adds new columns to the dataset, that correspond to the selected frequencies. Such columns are named with
    ``dft_{i}`` suffix. Suffix index do NOT indicate any relation to the frequencies. Produced names should be thought of as
    arbitrary identifiers to the produced sinusoids.
    """

    def __init__(self, k: int, in_column: str = "target", residuals: bool = False):
        """Init ``FourierDecomposeTransform``.

        Parameters
        ----------
        k:
            how many top positive frequencies selected for the decomposition. Selection performed proportional to the amplitudes.
        in_column:
            name of the processed column.
        residuals:
            whether to add residuals after decomposition. This guarantees that all components, including residuals, sum up to the series.
        """
        if k <= 0:
            raise ValueError("Parameter `k` must be positive integer!")

        self.k = k
        self.in_column = in_column
        self.residuals = residuals

        self._first_timestamp = None
        self._last_timestamp = None

        super().__init__(required_features=[in_column])

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def _fit(self, df: pd.DataFrame):
        """Fit transform with the dataframe."""
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform provided dataframe."""
        pass

    @staticmethod
    def _get_num_pos_freqs(series: pd.Series) -> int:
        """Get number of positive frequencies for the series."""
        num_obs = len(series)
        return int(np.ceil((num_obs - 1) / 2) + 1)

    def _check_segments(self, df: pd.DataFrame):
        """Check if series satisfy conditions."""
        segments_with_missing = []
        min_num_pos_freq = float("inf")
        for segment in df:
            series = df[segment]
            series = series.loc[series.first_valid_index() : series.last_valid_index()]
            if series.isna().any():
                segments_with_missing.append(segment)

            min_num_pos_freq = min(min_num_pos_freq, self._get_num_pos_freqs(series))

        if len(segments_with_missing) > 0:
            raise ValueError(
                f"Feature `{self.in_column}` contains missing values in segments: {segments_with_missing}!"
            )

        if self.k > min_num_pos_freq:
            raise ValueError(f"Parameter `k` must not be greater then {min_num_pos_freq} for the provided dataset!")

    def _dft_components(self, series: pd.Series) -> pd.DataFrame:
        """Estimate series decomposition using FFT."""
        initial_index = series.index
        series = series.loc[series.first_valid_index() : series.last_valid_index()]

        num_pos_freqs = self._get_num_pos_freqs(series)

        # compute Fourier decomposition of the series
        dft_series = np.fft.fft(series)

        # compute "amplitudes" for each frequency
        abs_dft_series = np.abs(dft_series)

        # select top-k indices
        abs_pos_dft_series = abs_dft_series[:num_pos_freqs]
        top_k_idxs = np.argpartition(abs_pos_dft_series, num_pos_freqs - self.k)[-self.k :]

        # select top-k and separate each frequency
        freq_matrix = np.diag(dft_series)
        freq_matrix = freq_matrix[:num_pos_freqs]
        selected_freqs = freq_matrix[top_k_idxs]

        # return frequencies to initial domain
        components = np.fft.ifft(selected_freqs).real

        components_df = pd.DataFrame(
            data=components.T, columns=[f"dft_{i}" for i in range(components.shape[0])], index=series.index
        )

        if self.residuals:
            components_df["dft_residuals"] = series.values - np.sum(components, axis=0)

        # return trailing and leading nans to the series if any existed initially
        if not components_df.index.equals(initial_index):
            components_df = components_df.reindex(index=initial_index, fill_value=np.nan)

        return components_df

    def fit(self, ts: TSDataset) -> "FourierDecomposeTransform":
        """Fit the transform and the decomposition model.

        Parameters
        ----------
        ts:
            dataset to fit the transform on.

        Returns
        -------
        :
            the fitted transform instance.
        """
        self._first_timestamp = ts.timestamps.min()
        self._last_timestamp = ts.timestamps.max()

        self._check_segments(df=ts[..., self.in_column].droplevel("feature", axis=1))

        return self

    def transform(self, ts: TSDataset) -> TSDataset:
        """Transform ``TSDataset`` inplace.

        Parameters
        ----------
        ts:
            Dataset to transform.

        Returns
        -------
        :
            Transformed ``TSDataset``.
        """
        if self._first_timestamp is None:
            raise ValueError("Transform is not fitted!")

        if ts.timestamps.min() < self._first_timestamp:
            raise ValueError(
                f"First index of the dataset to be transformed must be larger or equal than {self._first_timestamp}!"
            )

        if ts.timestamps.min() > self._last_timestamp:
            raise ValueError(
                f"Dataset to be transformed must contain historical observations in range {self._first_timestamp} - {self._last_timestamp}"
            )

        segment_df = ts[..., self.in_column].droplevel("feature", axis=1)

        ts_max_timestamp = ts.timestamps.max()
        if ts_max_timestamp > self._last_timestamp:
            future_steps = determine_num_steps(self._last_timestamp, ts_max_timestamp, freq=ts.freq)
            segment_df.iloc[-future_steps:] = np.nan

        self._check_segments(df=segment_df)

        segments = segment_df.columns
        segment_components = []
        for segment in segments:
            components_df = self._dft_components(series=segment_df[segment])
            components_df.columns = f"{self.in_column}_" + components_df.columns

            components_df.columns = pd.MultiIndex.from_product(
                [[segment], components_df.columns], names=["segment", "feature"]
            )

            segment_components.append(components_df)

        segment_components = pd.concat(segment_components, axis=1)

        columns_before = set(ts.features)
        columns_before &= set(segment_components.columns.get_level_values("feature"))
        self._update_dataset(ts=ts, columns_before=columns_before, df_transformed=segment_components)

        return ts


__all__ = ["FourierDecomposeTransform"]
