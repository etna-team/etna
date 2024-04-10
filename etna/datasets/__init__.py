"""Module with dataset and its utilities."""

from etna.datasets.datasets_generation import generate_ar_df
from etna.datasets.datasets_generation import generate_const_df
from etna.datasets.datasets_generation import generate_from_patterns_df
from etna.datasets.datasets_generation import generate_hierarchical_df
from etna.datasets.datasets_generation import generate_periodic_df
from etna.datasets.hierarchical_structure import HierarchicalStructure
from etna.datasets.internal_datasets import load_dataset
from etna.datasets.tsdataset import TSDataset
from etna.datasets.utils import DataFrameFormat
from etna.datasets.utils import apply_alignment
from etna.datasets.utils import duplicate_data
from etna.datasets.utils import infer_alignment
from etna.datasets.utils import make_timestamp_df_from_alignment
from etna.datasets.utils import set_columns_wide
