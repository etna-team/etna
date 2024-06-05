"""
MIT License

Copyright (c) 2022 Emadeldeen Eldele

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Note: Copied from ts-tcc repository (https://github.com/emadeldeen24/TS-TCC/tree/main)

import torch
from torch.utils.data import Dataset

from etna.libs.tstcc.augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(
            self,
            dataset,
            mode,
            jitter_scale_ratio,
            max_seg,
            jitter_ratio
    ):
        """
        Notes
        -----
        In this implementation we replace NaNs with 0 values to work with time-series with different length.
        """
        super(Load_Dataset, self).__init__()
        self.mode = mode
        self.jitter_scale_ratio = jitter_scale_ratio
        self.max_seg = max_seg
        self.jitter_ratio = jitter_ratio

        X_train = torch.from_numpy(dataset)
        X_train = torch.nan_to_num(X_train, nan=0)

        self.x_data = X_train
        self.len = X_train.shape[0]
        if self.mode == "train":
            aug1, aug2 = DataTransform(
                self.x_data,
                jitter_scale_ratio=self.jitter_scale_ratio,
                max_seg=self.max_seg,
                jitter_ratio=self.jitter_ratio
            )
            self.aug1, self.aug2 = aug1.float(), aug2.float()

    def __getitem__(self, index):
        if self.mode == "train":
            return self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index].float()

    def __len__(self):
        return self.len
