import pytest
import torch
from pandas.testing import assert_frame_equal

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.libs.chronos.chronos import ChronosModelForForecasting
from etna.libs.chronos.chronos_bolt import ChronosBoltModelForForecasting
from etna.models.nn import ChronosBoltModel
from etna.models.nn import ChronosModel
from etna.pipeline import Pipeline

df = generate_ar_df(start_time="2001-01-01", periods=10, n_segments=2)
df["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
ts = TSDataset(df, freq="D")

model = ChronosBoltModel(model_name="chronos-bolt-tiny", encoder_length=10, device="mps", dtype=torch.float32)
pipeline = Pipeline(model=model, horizon=1)
pipeline.fit(ts)
forecast = pipeline.forecast()
