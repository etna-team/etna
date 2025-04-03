from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from statsmodels.tsa.arima_process import arma_generate_sample

from etna.datasets.utils import _check_timestamp_param
from etna.datasets.utils import timestamp_range

_FREQ_DAY = pd.offsets.Day()


def _create_timestamp(
    start_time: Optional[Union[pd.Timestamp, int, str]], freq: Union[pd.DateOffset, str, None], periods: int
) -> Sequence[Union[pd.Timestamp, int]]:
    if freq is None and start_time is None:
        start_time = 0
    if freq is not None and start_time is None:
        start_time = pd.Timestamp("2000-01-01")
    _check_timestamp_param(param=start_time, param_name="start_time", freq=freq)
    return timestamp_range(start=start_time, periods=periods, freq=freq)


def generate_ar_df(
    periods: int,
    start_time: Optional[Union[pd.Timestamp, int, str]] = None,
    ar_coef: Optional[list] = None,
    sigma: float = 1,
    n_segments: int = 1,
    freq: Union[pd.DateOffset, str, None] = _FREQ_DAY,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with AR process data.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    ar_coef:
        AR coefficients
    sigma:
        scale of AR noise
    n_segments:
        number of segments
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp
    random_seed:
        random seed

    Raises
    ------
    ValueError:
        Incorrect type of ``start_time`` is used according to ``freq``
    """
    if ar_coef is None:
        ar_coef = [1]
    random_sampler = RandomState(seed=random_seed).normal
    ar_coef = np.r_[1, -np.array(ar_coef)]
    ar_samples = arma_generate_sample(
        ar=ar_coef, ma=[1], nsample=(n_segments, periods), axis=1, distrvs=random_sampler, scale=sigma
    )
    df = pd.DataFrame(data=ar_samples.T, columns=[f"segment_{i}" for i in range(n_segments)])
    df["timestamp"] = _create_timestamp(start_time=start_time, freq=freq, periods=periods)
    df = df.melt(id_vars=["timestamp"], value_name="target", var_name="segment")
    return df


def generate_periodic_df(
    periods: int,
    start_time: Optional[Union[pd.Timestamp, int, str]] = None,
    scale: float = 10,
    period: int = 1,
    n_segments: int = 1,
    freq: Union[pd.DateOffset, str, None] = _FREQ_DAY,
    add_noise: bool = False,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with periodic data.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    scale:
        we sample data from Uniform[0, scale)
    period:
        data frequency -- x[i+period] = x[i]
    n_segments:
        number of segments
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp
    add_noise:
        if True we add noise to final samples
    sigma:
        scale of added noise
    random_seed:
        random seed

    Raises
    ------
    ValueError:
        Non-integer timestamp parameter is used for integer-indexed timestamp.
    """
    samples = RandomState(seed=random_seed).randint(int(scale), size=(n_segments, period))
    patterns = [list(ar) for ar in samples]
    df = generate_from_patterns_df(
        periods=periods,
        start_time=start_time,
        patterns=patterns,
        sigma=sigma,
        random_seed=random_seed,
        freq=freq,
        add_noise=add_noise,
    )
    return df


def generate_const_df(
    periods: int,
    start_time: Optional[Union[pd.Timestamp, int, str]] = None,
    scale: float = 10,
    n_segments: int = 1,
    freq: Union[pd.DateOffset, str, None] = _FREQ_DAY,
    add_noise: bool = False,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with const data.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    scale:
        const value to fill
    n_segments:
        number of segments
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp
    add_noise:
        if True we add noise to final samples
    sigma:
        scale of added noise
    random_seed:
        random seed

    Raises
    ------
    ValueError:
        Non-integer timestamp parameter is used for integer-indexed timestamp.
    """
    patterns = [[scale] for _ in range(n_segments)]
    df = generate_from_patterns_df(
        periods=periods,
        start_time=start_time,
        patterns=patterns,
        sigma=sigma,
        random_seed=random_seed,
        freq=freq,
        add_noise=add_noise,
    )
    return df


def generate_from_patterns_df(
    periods: int,
    start_time: Optional[Union[pd.Timestamp, int, str]],
    patterns: List[List[float]],
    freq: Union[pd.DateOffset, str, None] = _FREQ_DAY,
    add_noise=False,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame from patterns.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    patterns:
        list of lists with patterns to be repeated
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp
    add_noise:
        if True we add noise to final samples
    sigma:
        scale of added noise
    random_seed:
        random seed

    Raises
    ------
    ValueError:
        Incorrect type of ``start_time`` is used according to ``freq``
    """
    n_segments = len(patterns)
    if add_noise:
        noise = RandomState(seed=random_seed).normal(scale=sigma, size=(n_segments, periods))
    else:
        noise = np.zeros(shape=(n_segments, periods))
    samples = noise
    for idx, pattern in enumerate(patterns):
        samples[idx, :] += np.array(pattern * (periods // len(pattern) + 1))[:periods]
    df = pd.DataFrame(data=samples.T, columns=[f"segment_{i}" for i in range(n_segments)])
    df["timestamp"] = _create_timestamp(start_time=start_time, freq=freq, periods=periods)
    df = df.melt(id_vars=["timestamp"], value_name="target", var_name="segment")
    return df


def generate_hierarchical_df(
    periods: int,
    n_segments: List[int],
    freq: Union[pd.DateOffset, str, None] = _FREQ_DAY,
    start_time: Optional[Union[pd.Timestamp, int, str]] = None,
    ar_coef: Optional[list] = None,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with hierarchical structure and AR process data.

    The hierarchical structure is generated as follows:
        1. Number of levels in the structure is the same as length of ``n_segments`` parameter
        2. Each level contains the number of segments set in ``n_segments``
        3. Connections from parent to child level are generated randomly.

    Parameters
    ----------
    periods:
        number of timestamps
    n_segments:
        number of segments on each level.
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp
    start_time:
        start timestamp
    ar_coef:
        AR coefficients
    sigma:
        scale of AR noise
    random_seed:
        random seed

    Returns
    -------
    :
        DataFrame at the bottom level of the hierarchy

    Raises
    ------
    ValueError:
        ``n_segments`` is empty
    ValueError:
        ``n_segments`` contains not positive integers
    ValueError:
        ``n_segments`` represents not non-decreasing sequence
    ValueError:
        Non-integer timestamp parameter is used for integer-indexed timestamp.
    """
    if len(n_segments) == 0:
        raise ValueError("`n_segments` should contain at least one positive integer!")

    if (np.less_equal(n_segments, 0)).any():
        raise ValueError("All `n_segments` elements should be positive!")

    if (np.diff(n_segments) < 0).any():
        raise ValueError("`n_segments` should represent non-decreasing sequence!")

    rnd = RandomState(seed=random_seed)

    bottom_df = generate_ar_df(
        periods=periods,
        start_time=start_time,
        ar_coef=ar_coef,
        sigma=sigma,
        n_segments=n_segments[-1],
        freq=freq,
        random_seed=random_seed,
    )

    bottom_segments = np.unique(bottom_df["segment"])

    n_levels = len(n_segments)
    child_to_parent = dict()
    for level_id in range(1, n_levels):
        prev_level_n_segments = n_segments[level_id - 1]
        cur_level_n_segments = n_segments[level_id]

        # ensure all parents have at least one child
        seen_ids = set()
        child_ids = rnd.choice(cur_level_n_segments, prev_level_n_segments, replace=False)
        for parent_id, child_id in enumerate(child_ids):
            seen_ids.add(child_id)
            child_to_parent[f"l{level_id}s{child_id}"] = f"l{level_id - 1}s{parent_id}"

        for child_id in range(cur_level_n_segments):
            if child_id not in seen_ids:
                parent_id = rnd.choice(prev_level_n_segments, 1).item()
                child_to_parent[f"l{level_id}s{child_id}"] = f"l{level_id - 1}s{parent_id}"

    bottom_segments_map = {segment: f"l{n_levels - 1}s{idx}" for idx, segment in enumerate(bottom_segments)}
    bottom_df[f"level_{n_levels - 1}"] = bottom_df["segment"].map(lambda x: bottom_segments_map[x])

    for level_id in range(n_levels - 2, -1, -1):
        bottom_df[f"level_{level_id}"] = bottom_df[f"level_{level_id + 1}"].map(lambda x: child_to_parent[x])

    bottom_df.drop(columns=["segment"], inplace=True)

    return bottom_df
