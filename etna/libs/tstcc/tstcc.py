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
# Moved training and encoding parameters from __init__ to fit and encode, respectively
# Moved device, batch_size, num_workers parameters from __init__ to fit and encode methods
# Changed input and output data shapes

from etna.libs.tstcc.encoder import ConvEncoder
from etna.libs.tstcc.tc import TC
from etna.libs.tstcc.dataloader import Load_Dataset
from etna.libs.tstcc.loss import NTXentLoss
from etna.loggers import tslogger
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TSTCC:
    '''TS-TCC model'''
    def __init__(
            self,
            input_dims,
            output_dims,
            kernel_size,
            dropout,
            timesteps,
            tc_hidden_dim,
            heads,
            depth,
            n_seq_steps,
            jitter_scale_ratio,
            max_seg,
            jitter_ratio,
            use_cosine_similarity
    ):
        """
        Init TSTCC model

        Parameters
        ----------
        input_dims:
            The input dimension. For a univariate time series, this should be set to 1.
        output_dims:
            The output dimension after encoder.
        kernel_size:
            Kernel size of first convolution in encoder.
        dropout:
            Dropout rate in first convolution block in encoder.
        timesteps:
            The number of timestamps to predict in temporal contrasting model.
        tc_hidden_dim:
            The output dimension after temporal_contr_model.
        heads:
            Number of heads in attention block in temporal contrasting model. Parameter output_dims must be a multiple
            of the number of heads.
        depth:
            Depth in attention block in temporal contrasting model.
        n_seq_steps:
            Max context size in temporal contrasting model.
        jitter_scale_ratio:
            Jitter ratio in weak augmentation.
        max_seg:
            Number of segments in strong augmentation.
        jitter_ratio:
            Jitter ratio in strong augmentation.
        use_cosine_similarity:
            If True NTXentLoss uses cosine similarity, if False NTXentLoss uses dot product.
        """

        super().__init__()

        self.input_dims = input_dims

        self.n_seq_steps = n_seq_steps

        self.model = torch.nn.ModuleDict({
                "encoder": ConvEncoder(
                    input_dims=self.input_dims,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    output_dims=output_dims
                ),
                "temporal_contr_model": TC(
                    input_dims=output_dims,
                    timesteps=timesteps,
                    hidden_dim=tc_hidden_dim,
                    heads=heads,
                    depth=depth,
                    n_seq_steps=self.n_seq_steps
                )
        })

        self.jitter_scale_ratio = jitter_scale_ratio
        self.max_seg = max_seg
        self.jitter_ratio = jitter_ratio

        self.use_cosine_similarity = use_cosine_similarity

    def prepare_data(self, data, mode, num_workers, batch_size):
        data = data.transpose(0, 2, 1)
        dataset = Load_Dataset(
            dataset=data,
            mode=mode,
            jitter_scale_ratio=self.jitter_scale_ratio,
            max_seg=self.max_seg,
            jitter_ratio=self.jitter_ratio
        )
        if mode == "train":
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers
            )
        else:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers
            )
        return data_loader

    def fit(self, train_data, n_epochs, lr, temperature, lambda1, lambda2, verbose, device, num_workers, batch_size):
        """
        Fit model

        Parameters
        ----------
        train_data:
            train data
        n_epochs:
            The number of epochs. When this reaches, the training stops.
        lr:
            The learning rate.
        temperature:
            Temperature in NTXentLoss.
        lambda1:
            The relative weight of the first item in the loss (temporal contrasting loss).
        lambda2:
            The relative weight of the second item in the loss (contextual contrasting loss).
        verbose:
            Whether to print the training loss after each epoch.
        device:
            The device used for training and inference.
        num_workers:
            How many subprocesses to use for data loading.
        batch_size:
            The batch size.
        """
        train_loader = self.prepare_data(data=train_data, mode="train", num_workers=num_workers, batch_size=batch_size)
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99),
                                           weight_decay=3e-4)
        self.model.to(device=device)
        self.model.train()
        for epoch in range(n_epochs):

            total_loss = []
            for batch_idx, (aug1, aug2) in enumerate(train_loader):
                # send to device
                aug1, aug2 = aug1.to(device), aug2.to(device)

                # optimizer
                model_optimizer.zero_grad()

                features1 = self.model.encoder(aug1)
                features2 = self.model.encoder(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_lstm_feat1 = self.model.temporal_contr_model(features1, features2, device)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.model.temporal_contr_model(features2, features1, device)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1
                zjs = temp_cont_lstm_feat2

                # compute loss
                nt_xent_criterion = NTXentLoss(
                    device=device,
                    batch_size=batch_size,
                    temperature=temperature,
                    use_cosine_similarity=self.use_cosine_similarity
                )
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

                total_loss.append(loss.item())
                loss.backward()
                model_optimizer.step()

            train_loss = torch.tensor(total_loss).mean()
            if verbose:
                tslogger.log(f"Epoch {epoch}: loss={train_loss:.4f}")

    def encode(self, data, encode_full_series, device, num_workers, batch_size):
        """
        Encode data

        Parameters
        ----------
        data:
            data to encode
        encode_full_series:
            if True the entire segment will be encoded.
        device:
            The device used for training and inference.
        num_workers:
            How many subprocesses to use for data loading.
        batch_size:
            The batch size.
        """
        data_loader = self.prepare_data(data=data, mode="encode", num_workers=num_workers, batch_size=batch_size)

        self.model.to(device=device)
        self.model.eval()

        embeddings = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                features = self.model.encoder(data)

                # normalize projection feature vectors
                features = F.normalize(features, dim=1)

                embeddings.append(features.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        if encode_full_series:
            embeddings = F.max_pool1d(embeddings, kernel_size=embeddings.shape[2],).squeeze(2)
        else:
            embeddings = embeddings.movedim(1, 2)
        return embeddings.numpy()

    def save(self, fn):
        ''' Save the model to a file.

        Args:
            fn_enc (str): filename
        '''
        torch.save(self.model.state_dict(), fn)

    def load(self, fn):
        ''' Load the model from a file.

        Args:
            fn_enc (str): filename
        '''
        state_dict = torch.load(fn, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
