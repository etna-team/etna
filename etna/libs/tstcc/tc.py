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
# Added ignoring warning about empty linear layer in self.projection_head when input_dims < 4

import warnings

import torch
import torch.nn as nn
import numpy as np
from etna.libs.tstcc.attention import Seq_Transformer


class TC(nn.Module):
    def __init__(
            self,
            input_dims,
            timesteps,
            hidden_dim,
            heads,
            depth,
            n_seq_steps
    ):
        super(TC, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = input_dims
        self.timestep = timesteps
        self.heads = heads
        self.depth = depth
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.n_seq_steps = n_seq_steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, input_dims // 2),
                nn.BatchNorm1d(input_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(input_dims // 2, input_dims // 4),
            )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=self.hidden_dim, depth=self.depth,
                                               heads=self.heads, mlp_dim=64)

    def forward(self, features_aug1, features_aug2, device):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(
            device)  # randomly pick time stamps

        score = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)

        forward_seq = z_aug1[:, max(0, t_samples - self.n_seq_steps):t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            score += torch.sum(torch.diag(self.lsoftmax(total)))
        score /= -1. * batch * self.timestep
        return score, self.projection_head(c_t)
