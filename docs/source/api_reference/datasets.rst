.. _datasets:

Datasets
========

.. automodule:: etna.datasets
    :no-members:
    :no-inherited-members:

API details
-----------

.. currentmodule:: etna.datasets

Basic structures:

.. autosummary::
   :toctree: api/
   :template: class.rst

   TSDataset
   DataFrameFormat
   HierarchicalStructure

Utilities for dataset generation:

.. autosummary::
   :toctree: api/
   :template: base.rst

   generate_ar_df
   generate_const_df
   generate_from_patterns_df
   generate_hierarchical_df
   generate_periodic_df

Utilities for data manipulation:

.. autosummary::
   :toctree: api/
   :template: base.rst

   duplicate_data
   set_columns_wide
   infer_alignment
   apply_alignment
   make_timestamp_df_from_alignment

Utilities for working with internal datasets:

.. autosummary::
   :toctree: api/
   :template: base.rst

   load_dataset
