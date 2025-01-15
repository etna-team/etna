<div align="center">
<img alt="Logo" src="https://github.com/etna-team/etna/raw/master/etna_logo.png" width="100%"/>
</div>

<h3 align="center">Predict your time series the easiest way</h3>

<p align="center">
  <a href="https://pypi.org/project/etna/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/etna.svg" /></a>
  <a href="https://pypi.org/project/etna/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/etna.svg" /></a>
  <a href="https://pepy.tech/project/etna"><img alt="Downloads" src="https://static.pepy.tech/personalized-badge/etna?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads" /></a>
</p>
 
<p align="center">
  <a href="https://codecov.io/gh/etna-team/etna"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/etna-team/etna.svg" /></a>
  <a href="https://github.com/etna-team/etna/actions/workflows/test.yml?query=branch%3Amaster++"><img alt="Test passing" src="https://img.shields.io/github/actions/workflow/status/etna-team/etna/test.yml?branch=master&label=tests" /></a>
  <a href="https://github.com/etna-team/etna/actions/workflows/publish.yml"><img alt="Docs publish" src="https://img.shields.io/github/actions/workflow/status/etna-team/etna/publish.yml?label=docs" /></a>
  <a href="https://github.com/etna-team/etna/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/etna-team/etna.svg" /></a>
</p>

<p align="center">
  <a href="https://t.me/etna_support"><img alt="Telegram" src="https://img.shields.io/badge/channel-telegram-blue" /></a>
  <a href="https://github.com/etna-team/etna/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/etna-team/etna.svg" /></a>
  <a href="https://github.com/etna-team/etna/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/etna-team/etna?style=social" /></a>
</p>
  
<p align="center">
  <a href="https://etna.tinkoff.ru">Homepage</a> | 
  <a href="https://docs.etna.ai/stable/">Documentation</a> |
  <a href="https://github.com/etna-team/etna/tree/master/examples">Tutorials</a> |
  <a href="https://github.com/etna-team/etna/blob/master/CONTRIBUTING.md">Contribution Guide</a> |
  <a href="https://github.com/etna-team/etna/releases">Release Notes</a>
</p>

ETNA is an easy-to-use time series forecasting framework. 
It includes built in toolkits for time series preprocessing, feature generation, 
a variety of predictive models with unified interface - from classic machine learning
to SOTA neural networks, models combination methods and smart backtesting.
ETNA is designed to make working with time series simple, productive, and fun. 

ETNA is the first python open source framework of 
[T-Bank.ru](https://www.tbank.ru)
Artificial Intelligence Center. 
The library started as an internal product in our company - 
we use it in over 10+ projects now, so we often release updates. 
Contributions are welcome - check our [Contribution Guide](https://github.com/etna-team/etna/blob/master/CONTRIBUTING.md).

## Quickstart

Let's load and prepare the data.
```python
import pandas as pd
from etna.datasets import TSDataset

# Read the data
df = pd.read_csv("examples/data/example_dataset.csv")

# Create a TSDataset
ts = TSDataset(df, freq="D")

# Choose a horizon
HORIZON = 14

# Make train/test split
train_ts, test_ts = ts.train_test_split(test_size=HORIZON)
```

Define transformations and model:
```python
from etna.models import CatBoostMultiSegmentModel
from etna.transforms import DateFlagsTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import FourierTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import MeanTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import TimeSeriesImputerTransform
from etna.transforms import TrendTransform

# Prepare transforms
transforms = [
    DensityOutliersTransform(in_column="target", distance_coef=3.0),
    TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),
    LinearTrendTransform(in_column="target"),
    TrendTransform(in_column="target", out_column="trend"),
    LagTransform(in_column="target", lags=list(range(HORIZON, 122)), out_column="target_lag"),
    DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
    FourierTransform(period=360.25, order=6, out_column="fourier"),
    SegmentEncoderTransform(),
    MeanTransform(in_column=f"target_lag_{HORIZON}", window=12, seasonality=7),
    MeanTransform(in_column=f"target_lag_{HORIZON}", window=7),
]

# Prepare model
model = CatBoostMultiSegmentModel()
```

Fit `Pipeline` and make a prediction:
```python
from etna.pipeline import Pipeline

# Create and fit the pipeline
pipeline = Pipeline(model=model, transforms=transforms, horizon=HORIZON)
pipeline.fit(train_ts)

# Make a forecast
forecast_ts = pipeline.forecast()
```

Let's plot the results:
```python
from etna.analysis import plot_forecast

plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts, n_train_samples=50)
```

![](examples/assets/readme/quickstart.png)

Print the metric value across the segments:
```python
from etna.metrics import SMAPE

metric = SMAPE()
metric_value = metric(y_true=test_ts, y_pred=forecast_ts)
metric_value
```

```bash
{'segment_a': 4.799114474387907, 'segment_b': 3.271014290441896, 'segment_c': 6.758606238307858, 'segment_d': 4.512871862697337}
```

Notebook with this example is available [here](examples/quickstart.ipynb).

## Installation 

ETNA is available on [PyPI](https://pypi.org/project/etna), so you can use `pip` to install it.

Install default version:
```bash
pip install --upgrade pip
pip install etna
```

The default version doesn't contain all the dependencies, because some of them are needed only for specific models, e.g. Prophet, PyTorch.
Available user extensions are the following:
* `prophet`: adds prophet model`,
* `torch`: adds models based on neural nets,
* `wandb`: adds wandb logger,
* `auto`: adds AutoML functionality,
* `statsforecast`: adds models from [statsforecast](https://nixtla.github.io/statsforecast/),
* `classiciation`: adds time series classification functionality,
* `chronos`: adds Chronos-like pretrained models,
* `timesfm`: adds TimesFM pretrained models.

Install extension:
```bash
pip install etna[extension-name]
```

Install all the extensions:
```bash
pip install etna[all]
```

There are also developer extensions. All the extensions are listed in [`pyproject.toml`](https://github.com/etna-team/etna/blob/master/pyproject.toml#L93).

Without the appropriate extension you will get an `ImportError` trying to import the model that needs it.
For example, `etna.models.ProphetModel` needs `prophet` extension and can't be used without it.

### Configuration

ETNA supports configuration files. It means that library will check that all the specified packages are installed prior to script start and NOT during runtime. 

To set up a configuration for your project you should create a `.etna` file at the project's root. To see the available options look at [`Settings`](https://github.com/etna-team/etna/blob/master/etna/settings.py#L94). There is an [example](https://github.com/etna-team/etna/tree/master/examples/configs/.etna) of configuration file.

## Tutorials

We have also prepared a set of tutorials for an easy introduction:

| Notebook                                                                                                                      |                                                                                                                                          Interactive launch |
|:------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [Get started](https://github.com/etna-team/etna/tree/master/examples/101-get_started.ipynb)                                   |                  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/101-get_started.ipynb) |
| [Backtest](https://github.com/etna-team/etna/tree/master/examples/102-backtest.ipynb)                                         |                     [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/102-backtest.ipynb) |
| [EDA](https://github.com/etna-team/etna/tree/master/examples/103-EDA.ipynb)                                                   |                          [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/103-EDA.ipynb) |
| [Regressors and exogenous data](https://github.com/etna-team/etna/tree/master/examples/201-exogenous_data.ipynb)              |               [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/201-exogenous_data.ipynb) |
| [Deep learning models](https://github.com/etna-team/etna/tree/master/examples/202-NN_examples.ipynb)                          |                  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/202-NN_examples.ipynb) |
| [Ensembles](https://github.com/etna-team/etna/tree/master/examples/303-ensembles.ipynb)                                       |                    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/203-ensembles.ipynb) |
| [Outliers](https://github.com/etna-team/etna/tree/master/examples/204-outliers.ipynb)                                         |                     [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/204-outliers.ipynb) |
| [AutoML](https://github.com/etna-team/etna/tree/master/examples/205-automl.ipynb)                                             |                       [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/205-automl.ipynb) |
| [Clustering](https://github.com/etna-team/etna/tree/master/examples/206-clustering.ipynb)                                     |                   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/206-clustering.ipynb) |
| [Feature selection](https://github.com/etna-team/etna/blob/master/examples/207-feature_selection.ipynb)                       |            [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/207-feature_selection.ipynb) |
| [Forecasting strategies](https://github.com/etna-team/etna/tree/master/examples/208-forecasting_strategies.ipynb)             |       [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/208-forecasting_strategies.ipynb) |
| [Mechanics of forecasting](https://github.com/etna-team/etna/blob/master/examples/209-mechanics_of_forecasting.ipynb)         |     [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/209-mechanics_of_forecasting.ipynb) |
| [Embedding models](https://github.com/etna-team/etna/blob/master/examples/210-embedding_models.ipynb)                         |             [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/210-embedding_models.ipynb) |
| [Custom model and transform](https://github.com/etna-team/etna/tree/master/examples/301-custom_transform_and_model.ipynb)     |   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/301-custom_transform_and_model.ipynb) |
| [Inference: using saved pipeline on a new data](https://github.com/etna-team/etna/tree/master/examples/302-inference.ipynb)   |                    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/302-inference.ipynb) |
| [Hierarchical time series](https://github.com/etna-team/etna/blob/master/examples/303-hierarchical_pipeline.ipynb)            |        [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/303-hierarchical_pipeline.ipynb) |
| [Forecast interpretation](https://github.com/etna-team/etna/tree/master/examples/304-forecasting_interpretation.ipynb)        |   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/304-forecasting_interpretation.ipynb) |
| [Classification](https://github.com/etna-team/etna/blob/master/examples/305-classification.ipynb)                             |               [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/305-classification.ipynb) |
| [Prediction intervals](https://github.com/etna-team/etna/tree/master/examples/306-prediction_intervals.ipynb)                 |         [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/306-prediction_intervals.ipynb) |
| [Working with misaligned data](https://github.com/etna-team/etna/tree/master/examples/307-working_with_misaligned_data.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/etna-team/etna/master?filepath=examples/307-working_with_misaligned_data.ipynb) |

## Documentation

ETNA documentation is available [here](https://docs.etna.ai/stable/).

## Community

Questions and feedback are welcome! Our channels for communication:
- [Discussions](https://github.com/etna-team/etna/discussions)
  - Suggestions with ideas and drawbacks
  - Q&A, e.g. usage questions
  - General discussions
- [Issue tracker](https://github.com/etna-team/etna/issues)
  - Bug reports
  - Tasks
- [Telegram chat](https://t.me/etna_support)
  - Useful for any other form of communication

## Resources

- [Forecasting using ETNA library | 60 lines Catboost](https://www.kaggle.com/code/goolmonika/forecasting-using-etna-library-60-lines-catboost) on Kaggle

- [Прикладные задачи анализа данных, лекция 8 — Временные ряды 2](https://youtu.be/1gXVbidDZck) on YouTube

- [Time Series Forecasting Strategies in ETNA](https://medium.com/its-tinkoff/time-series-forecasting-strategies-in-etna-93d7d2f8a911) on Medium

- [ETNA: Time Series Analysis. What, why and how?](https://medium.com/its-tinkoff/etna-time-series-analysis-what-why-and-how-e45557af4f6c) on Medium

- [ETNA Meetup Jun 2022](https://www.youtube.com/watch?v=N1Xy3EqY058&list=PLLrf_044z4JrSsjMd-3dF6VbBLPI_yOxG) on YouTube

- [DUMP May 2022 talk](https://youtu.be/12uuxepdtks) on YouTube

- [ETNA Regressors](https://medium.com/its-tinkoff/etna-regressors-d2722923e88e) on Medium

- [Time series forecasting with ETNA: first steps](https://medium.com/its-tinkoff/time-series-forecasting-with-etna-first-steps-dfaf90c5b919) on Medium

- [EDA notebook for Ubiquant Market Prediction](https://www.kaggle.com/code/martins0n/ubiquant-eda-toy-predictions-etna) on Kaggle

- [Tabular Playground Series - Mar 2022 (7th place!)](https://www.kaggle.com/code/chikovalexander/tps-mar-2022-etna/notebook?scriptVersionId=91575908) on Kaggle

- [Tabular Playground Series - Jan 2022](https://www.kaggle.com/code/chikovalexander/tps-jan-2022-etna/notebook) on Kaggle

- [Forecasting with ETNA: Fast and Furious](https://medium.com/its-tinkoff/forecasting-with-etna-fast-and-furious-1b58e1453809) on Medium

- [Store sales prediction with etna library](https://www.kaggle.com/dmitrybunin/store-sales-prediction-with-etna-library?scriptVersionId=81104235) on Kaggle

- [PyCon Russia September 2021 talk](https://youtu.be/VxWHLEFgXnE) on YouTube

## Acknowledgments

### ETNA.Team

Current team members:
[Dmitriy Bunin](https://github.com/Mr-Geekman),
[Aleksandr Chikov](https://github.com/alex-hse-repository),
[Vladislav Denisov](https://github.com/v-v-denisov),
[Martin Gabdushev](https://github.com/martins0n),
[Artem Makhin](https://github.com/Ama16),
[Ivan Mitskovets](https://github.com/imitskovets),
[Albina Munirova](https://github.com/albinamunirova),
[Ivan Nedosekov](https://github.com/GrozniyToaster),
[Rodion Petrov](https://github.com/Noidor1),
[Maxim Zherelo](https://github.com/brsnw250),
[Yakov Malyshev](https://github.com/ostreech1997),
[Egor Baturin](https://github.com/egoriyaa),
[Mikhail Bolev](https://github.com/kenshi777),
[Danil Smorchkov](https://github.com/DanilSmorchkov),

Former team members:
[Andrey Alekseev](https://github.com/iKintosh),
[Nikita Barinov](https://github.com/diadorer),
[Julia Shenshina](https://github.com/julia-shenshina),
[Sergey Kolesnikov](https://github.com/Scitator),
[Yuriy Tarasyuk](https://github.com/DBcreator),
[Konstantin Vedernikov](https://github.com/scanhex12),
[Nikolai Romantsov](https://github.com/WinstonDovlatov),
[Sergei Zhuravlev](https://github.com/malodetz),
[Alexandr Kuznetsov](https://github.com/Alexander76Kuznetsov),
[Grigory Zlotin](https://github.com/yellowssnake),
[Dmitriy Sablin](https://github.com/Polzovat123),
[Artem Levashov](https://github.com/soft1q),
[Aleksey Podkidyshev](https://github.com/alekseyen)

### ETNA.Contributors

[GooseIt](https://github.com/GooseIt),
[mvakhmenin](https://github.com/mvakhmenin),
[looopka](https://github.com/looopka),
[aleksander43smith](https://github.com/aleksander43smith),
[smetam](https://github.com/smetam),
[Wapwolf](https://github.com/Wapwolf),
[ArtemLiA](https://github.com/ArtemLiA),
[Carlosbogo](https://github.com/Carlosbogo),
[GoshaLetov](https://github.com/GoshaLetov),
[LeorFinkelberg](https://github.com/LeorFinkelberg),
[Pacman1984](https://github.com/Pacman1984),

## License

Feel free to use our library in your commercial and private applications.

ETNA is covered by [Apache 2.0](/LICENSE). 
Read more about this license [here](https://choosealicense.com/licenses/apache-2.0/)

> Please note that `etna[prophet]` is covered by [GPL 2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) due to pystan package.
