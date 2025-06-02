from enum import Enum
from typing import Any
from typing import Dict
from typing import List

from hydra_slayer import get_from_params

from etna.auto.pool import templates
from etna.auto.pool.utils import fill_template
from etna.pipeline import Pipeline


class PoolGenerator:
    """Generate a pool of pipelines from given config templates in hydra format."""

    def __init__(self, configs_template: List[dict]):
        """
        Initialize with a list of config templates in hydra format.

        Parameters
        ----------
        configs_template:
            list of template configs in hydra format

        Notes
        -----
        Hydra configs templates:
        ::
            {
                '_target_': 'etna.pipeline.Pipeline',
                'horizon': '${__aux__.horizon}',
                'model': {'_target_': 'etna.models.ProphetModel'}
            }
        Values to be interpolated should be in the form of ``${__aux__.key}``
        """
        self.configs_template = configs_template

    def generate(self, horizon: int, generate_params: Dict[str, Any]) -> List[Pipeline]:
        """
        Fill templates with args.

        Parameters
        ----------
        horizon:
            horizon to forecast
        generate_params
            Dictionary with parameters to fill pool templates.
        """
        params_to_fill = templates.REQUIRED_PARAMS | generate_params
        params_to_fill["horizon"] = horizon
        filled_templates: List[dict] = [fill_template(config, params_to_fill) for config in self.configs_template]
        return [get_from_params(**filled_template) for filled_template in filled_templates]


class Pool(Enum):
    """Predefined pools of pipelines.

    Pools are divided into types by frequency (``D`` (daily), ``H`` (hourly), ``MS`` (monthly), ``W`` (weekly), ``no_freq``) and
    duration (``super_fast``, ``fast``, ``medium`` and ``heavy``).

    Division by frequency:

    - The ``D`` group of pools should be chosen in a case of "D" frequency.
    - The ``H`` group of pools should be chosen in a case of "h" frequency.
    - The ``MS`` group of pools should be chosen in a case of "ME" and "MS" frequencies.
    - The ``W`` group of pools should be chosen in a case of following frequencies: "W", "W-MON", "W-SUN", ..., etc.
    - The ``no_freq`` group of pools should be chosen in a case of other frequencies, frequencies with the gap between timestamps more than 1 (T, Q, 2D, 4H, ..., etc) and series without datetime timestamps.

    Frequency in your data is defined by value of ``freq`` parameter passed to ``TSDataset`` constructor or ``TSDataset.create_from_misaligned`` method.

    Division by duration:

    Each subsequent pool is a subset of the next one: ``fast`` pool contains all configs from ``super_fast`` plus some other,
    ``medium`` contains all configs from ``fast`` plus some other and ``heavy`` contains all configs from ``medium`` plus some other.

    To get final pool choose one frequency and duration, for example ``D_super_fast``.

    Pools can contain ``timestamp_column``, ``chronos_device`` and ``timesfm_device`` parameters that can be filled by user.
    Default values are ``None``, ``auto`` and ``gpu`` respectively.

    - Parameter ``timestamp_column`` can be used starting from ``fast`` pools.
    - Parameter ``chronos_device`` can be used starting from ``super_fast`` pools.
    - Parameter ``timesfm_device`` can be used starting from ``fast`` pools.

    Note
    ----
    This class requires ``auto`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    no_freq_super_fast = PoolGenerator(configs_template=templates.NO_FREQ_SUPER_FAST)  # type: ignore
    no_freq_fast = PoolGenerator(configs_template=templates.NO_FREQ_FAST)  # type: ignore
    no_freq_medium = PoolGenerator(configs_template=templates.NO_FREQ_MEDIUM)  # type: ignore
    no_freq_heavy = PoolGenerator(configs_template=templates.NO_FREQ_HEAVY)  # type: ignore

    D_super_fast = PoolGenerator(configs_template=templates.D_SUPER_FAST)  # type: ignore
    D_fast = PoolGenerator(configs_template=templates.D_FAST)  # type: ignore
    D_medium = PoolGenerator(configs_template=templates.D_MEDIUM)  # type: ignore
    D_heavy = PoolGenerator(configs_template=templates.D_HEAVY)  # type: ignore

    H_super_fast = PoolGenerator(configs_template=templates.H_SUPER_FAST)  # type: ignore
    H_fast = PoolGenerator(configs_template=templates.H_FAST)  # type: ignore
    H_medium = PoolGenerator(configs_template=templates.H_MEDIUM)  # type: ignore
    H_heavy = PoolGenerator(configs_template=templates.H_HEAVY)  # type: ignore

    MS_super_fast = PoolGenerator(configs_template=templates.MS_SUPER_FAST)  # type: ignore
    MS_fast = PoolGenerator(configs_template=templates.MS_FAST)  # type: ignore
    MS_medium = PoolGenerator(configs_template=templates.MS_MEDIUM)  # type: ignore
    MS_heavy = PoolGenerator(configs_template=templates.MS_HEAVY)  # type: ignore

    W_super_fast = PoolGenerator(configs_template=templates.W_SUPER_FAST)  # type: ignore
    W_fast = PoolGenerator(configs_template=templates.W_FAST)  # type: ignore
    W_medium = PoolGenerator(configs_template=templates.W_MEDIUM)  # type: ignore
    W_heavy = PoolGenerator(configs_template=templates.W_HEAVY)  # type: ignore
