.. _metrics:

Metrics
========

.. automodule:: etna.metrics
    :no-members:
    :no-inherited-members:

API details
-----------

.. currentmodule:: etna.metrics

Base:

.. autosummary::
   :toctree: api/
   :template: class.rst

   Metric
   MetricWithMissingHandling

Enums:

.. autosummary::
   :toctree: api/
   :template: class.rst

   MetricAggregationMode
   MetricMissingMode

Scalar metrics:

.. autosummary::
   :toctree: api/
   :template: class.rst

   MAE
   MAPE
   MSE
   MSLE
   R2
   RMSE
   SMAPE
   WAPE
   MaxDeviation
   MedAE
   Sign
   MissingCounter

Interval metrics:

.. autosummary::
   :toctree: api/
   :template: class.rst

   Coverage
   Width

Utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   compute_metrics
