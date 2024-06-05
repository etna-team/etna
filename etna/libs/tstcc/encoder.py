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

# In the original implementation, the name of this file is "model.py".
# Added ignoring warning about even kernel lengths and odd dilation in nn.Conv1d blocks.
import warnings

from torch import nn


class ConvEncoder(nn.Module):
    def __init__(
            self,
            input_dims,
            kernel_size,
            dropout,
            output_dims
    ):
        super(ConvEncoder, self).__init__()

        self.input_dims = input_dims
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.output_dims = output_dims

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_dims, 32, kernel_size=self.kernel_size,
                      stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, output_dims, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(output_dims),
            nn.ReLU(),
        )

    def forward(self, x_in):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.conv_block1(x_in)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
        return x
