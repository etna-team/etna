.. _internal_datasets:

Internal datasets
============

ETNA library contains several popular datasets that are often used in papers to estimate the quality of time series
models. To load them you choose dataset name and use the following code:

..  code-block:: python

    from etna.datasets import load_dataset

    ts = load_dataset(name="tourism_monthly", parts="full")


The first time, it will take some time to download and save the dataset locally, the next times the data will be read from a file.
In the example above, we load ``tourism`` dataset with monthly frequency. We also use ``parts="full"``, which means that we load
the full dataset (each dataset has predefined parts to load).

List of internal datasets
-------------------------

electricity dataset
^^^^^^^^^^^^^^^^^^^
The electricity dataset is a 15 minutes time series of electricity consumption (in kW)
of 370 customers. It has three parts:


Loading names:

- ``electricity_15T`` with parts: train (139896 observations), test (360 observations), full (140256 observations)

References:

- https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014


m3 dataset
^^^^^^^^^^
The M3 dataset is a collection of 3,003 time series used for the third edition of the Makridakis forecasting
Competition. The M3 dataset consists of time series of yearly, quarterly, monthly and other data. Dataset with other
data originally does not have any particular frequency, but we assume it as a quarterly data. Each frequency mode
has its own specific prediction horizon: 6 for yearly, 8 for quarterly, 18 for monthly, and 8 for other.

M3 dataset has series ending on different dates. As to the specificity of ``TSDataset`` we should add custom dates
to make series end on one date. Original dates are added as an exogenous data. For example, ``df_exog`` of train
dataset has dates for train and test and ``df_exog`` of test dataset has dates only for test.

Loading names:

- ``m3_monthly`` with parts: train (126 observations), test (18 observations), full (144 observations)
- ``m3_quarterly`` with parts: train (64 observations), test (8 observations), full (72 observations)
- ``m3_yearly`` with parts: train (41 observations), test (6 observations), full (47 observations)
- ``m3_other`` with parts: train (96 observations), test (8 observations), full (104 observations)

References:

- https://forvis.github.io/datasets/m3-data/
- https://forecasters.org/resources/time-series-data/m3-competition/


m4 dataset
^^^^^^^^^^
The M4 dataset is a collection of 100,000 time series used for the fourth edition of the Makridakis forecasting
Competition. The M4 dataset consists of time series of yearly, quarterly, monthly and other (weekly, daily and
hourly) data. Each frequency mode has its own specific prediction horizon: 6 for yearly, 8 for quarterly,
18 for monthly, 13 for weekly, 14 for daily and 48 for hourly.

Loading names:

- ``m4_hourly`` with parts: train (960 observations), test (48 observations), full (1008 observations)
- ``m4_daily`` with parts: train (9919 observations), test (14 observations), full (9933 observations)
- ``m4_weekly`` with parts: train (2597 observations), test (13 observations), full (2610 observations)
- ``m4_monthly`` with parts: train (2794 observations), test (18 observations), full (2812 observations)
- ``m4_quarterly`` with parts: train (866 observations), test (8 observations), full (874 observations)
- ``m4_yearly`` with parts: train (835 observations), test (6 observations), full (841 observations)

References:

- https://github.com/Mcompetitions/M4-methods


traffic 2008 dataset
^^^^^^^^^^^^^^^^^^^^
15 months worth of daily data (440 daily records) that describes the occupancy rate, between 0 and 1, of different
car lanes of the San Francisco bay area freeways across time. Data was collected by 963 sensors from
Jan. 1st 2008 to Mar. 30th 2009 (15 days were dropped from this period: public holidays and two days with
anomalies, we set zero values for these days). Initial dataset has 10 min frequency, we create traffic with hour
frequency by mean aggregation. Each frequency mode has its own specific prediction horizon: 6 * 24 for 10T,
24 for hourly.

Loading names:

- ``traffic_2008_10T`` with parts: train (65376 observations), test (144 observations), full (65520 observations)
- ``traffic_2008_hourly`` with parts: train (10896 observations), test (24 observations), full (10920 observations)

References:

- https://archive.ics.uci.edu/dataset/204/pems+sf
- http://pems.dot.ca.gov


traffic 2015 dataset
^^^^^^^^^^^^^^^^^^^^
24 months worth of hourly data (24 daily records) that describes the occupancy rate, between 0 and 1, of different
car lanes of the San Francisco bay area freeways across time. Data was collected by 862 sensors from
Jan. 1st 2015 to Dec. 31th 2016. Dataset has prediction horizon: 24.

Loading names:

- ``traffic_2015_hourly`` with parts: train (17520 observations), test (24 observations), full (17544 observations)

References:

- https://github.com/laiguokun/multivariate-time-series-data
- http://pems.dot.ca.gov


tourism dataset
^^^^^^^^^^^^^^^
Dataset contains 1311 series in three frequency modes: monthly, quarterly, yearly. They were supplied by both
tourism bodies (such as Tourism Australia, the Hong Kong Tourism Board and Tourism New Zealand) and various
academics, who had used them in previous tourism forecasting studies. Each frequency mode has its own specific
prediction horizon: 4 for yearly, 8 for quarterly, 24 for monthly.

Tourism dataset has series ending on different dates. As to the specificity of ``TSDataset`` we should add custom dates
to make series end on one date. Original dates are added as an exogenous data. For example, ``df_exog`` of train
dataset has dates for train and test and ``df_exog`` of test dataset has dates only for test.

Loading names:

- ``tourism_monthly`` with parts: train (309 observations), test (24 observations), full (333 observations)
- ``tourism_quarterly`` with parts: train (122 observations), test (8 observations), full (130 observations)
- ``tourism_yearly`` with parts: train (43 observations), test (4 observations), full (47 observations)

References:

- https://robjhyndman.com/publications/the-tourism-forecasting-competition/


weather dataset
^^^^^^^^^^^^^^^
Dataset contains 21 meteorological indicators in Germany, such as humidity and air temperature with a 10 min
frequency for 2020. We use the last 24 hours as prediction horizon.

Loading names:

- ``weather_10T`` with parts: train (52560 observations), test (144 observations), full (52704 observations)

References:

- https://www.bgc-jena.mpg.de/wetter/


Electricity Transformer Datasets (ETT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dataset consists of four parts: ETTh1 (hourly freq), ETTh2 (hourly freq), ETTm1 (15 min freq), ETTm2 (15 min freq).
This dataset is a collection of two years of data from two regions of a province of China. There are one target
column ("oil temperature") and six different types of external power load features. We use the last 720 hours as
prediction horizon.

Loading names:

- ``ETTm1`` with parts: train (66800 observations), test (2880 observations), full (69680 observations)
- ``ETTm2`` with parts: train (66800 observations), test (2880 observations), full (69680 observations)
- ``ETTh1`` with parts: train (16700 observations), test (720 observations), full (17420 observations)
- ``ETTh2`` with parts: train (16700 observations), test (720 observations), full (17420 observations)


References:

- https://www.bgc-jena.mpg.de/wetter/
- https://arxiv.org/abs/2012.07436


Individual household electric power consumption dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This dataset consists of almost 4 years of history with 1 minute frequency from a household in Sceaux. Different
electrical quantities and some sub-metering values are available.

Loading names:

- ``IHEPC_T`` with parts: full (2075259 observations)

References:

- https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

