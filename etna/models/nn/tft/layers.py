from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        """Init Gated Linear Unit.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        output_size:
            output size of the feature representation
        dropout:
            dropout rate
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.activation_fc = nn.Linear(self.input_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.gated_fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data with shapes (batch_size, num_timestamps, input_size)

        Returns
        -------
        :
            output batch of data with shapes (batch_size, num_timestamps, output_size)
        """
        x = self.dropout(x)
        a = self.activation_fc(x)
        b = self.sigmoid(self.gated_fc(x))
        x = torch.mul(a, b)
        return x


class GateAddNorm(nn.Module):
    """Gated Add&Norm layer."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        """Init Add&Norm layer.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        output_size:
            output size of the feature representation
        dropout:
            dropout rate
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(input_size=self.input_size, output_size=self.output_size, dropout=self.dropout)
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data with shapes (batch_size, num_timestamps, input_size)
        residual:
            batch of data passed through skip connection with shapes (batch_size, num_timestamps, output_size)
        Returns
        -------
        :
            output batch of data with shapes (batch_size, num_timestamps, output_size)
        """
        x = self.glu(x)
        x = self.norm(x + residual)
        return x


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        """Init GRN.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        output_size:
            output size of the feature representation
        dropout:
            dropout rate
        context_size:
            dimension of context vector
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size

        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context_fc = nn.Linear(self.context_size, self.input_size, bias=False)

        self.residual_fc = nn.Linear(self.input_size, self.output_size) if self.input_size != self.output_size else None
        self.fc2 = nn.Linear(self.input_size, self.input_size)

        self.gate_norm = GateAddNorm(
            input_size=self.input_size,
            output_size=self.output_size,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data with shapes (batch_size, num_timestamps, input_size)
        context:
            batch of data passed as the context through the block with shapes (batch_size, num_timestamps, output_size)
        Returns
        -------
        :
            output batch of data with shapes (batch_size, num_timestamps, output_size)
        """
        residual = self.residual_fc(x) if self.residual_fc is not None else x
        x = self.fc1(x)
        if context is not None:
            x = x + self.context_fc(context)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x=x, residual=residual)
        return x


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network."""

    def __init__(
        self,
        input_size: int,
        features: List[str],
        context_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Init Variable Selection Network.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        features:
            features to pass through the block
        context_size:
            dimension of context vector
        dropout:
            dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.features = features
        self.context_size = context_size
        self.dropout = dropout
        self.grns = nn.ModuleDict(
            {
                feature: GatedResidualNetwork(
                    input_size=self.input_size,
                    output_size=self.input_size,
                    dropout=self.dropout,
                )
                for feature in self.features
            }
        )
        self.flatten_grn = GatedResidualNetwork(
            input_size=self.input_size * self.num_features,
            output_size=self.num_features,
            dropout=self.dropout,
            context_size=self.context_size,
        )
        self.softmax = nn.Softmax(dim=2)

    @property
    def num_features(self) -> int:
        """Get number of all features.

        Returns
        -------
        :
            number of all features.
        """
        return len(self.features)

    def forward(self, x: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            dictionary where keys are feature names and values are transformed inputs for each feature
            with shapes (batch_size, num_timestamps, input_size)
        context:
            batch of data passed as the context through the block with shapes (batch_size, num_timestamps, num_features * input_size)
        Returns
        -------
        :
            output batch of data with shapes (batch_size, num_timestamps, output_size)
        """
        output = torch.zeros(list(x.values())[0].size() + torch.Size([len(x)])).to(
            list(x.values())[0].device
        )  # (batch_size, num_timestamps, input_size, num_features)
        for i, (feature, embedding) in enumerate(x.items()):
            output[:, :, :, i] = self.grns[feature](embedding)

        flatten_input = torch.cat(
            [x[feature] for feature in self.features], dim=-1
        )  # (batch_size, num_timestamps, input_size * num_features)
        flatten_grn_output = self.flatten_grn(
            x=flatten_input, context=context
        )  # (batch_size, num_timestamps, num_features)
        feature_weights = self.softmax(flatten_grn_output).unsqueeze(
            dim=-2
        )  # (batch_size, num_timestamps, 1, num_features)

        output = (output * feature_weights).sum(dim=-1)  # (batch_size, num_timestamps, input_size)
        return output


class StaticCovariateEncoder(nn.Module):
    """Static Covariate Encoder."""

    def __init__(self, input_size: int, dropout: float = 0.1):
        """Init Static Covariate Encoder.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        dropout:
            dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.grn_s = GatedResidualNetwork(  # for VariableSelectionNetwork
            input_size=self.input_size, output_size=self.input_size, dropout=self.dropout
        )
        self.grn_c = GatedResidualNetwork(  # for LSTM
            input_size=self.input_size, output_size=self.input_size, dropout=self.dropout
        )
        self.grn_h = GatedResidualNetwork(  # for LSTM
            input_size=self.input_size, output_size=self.input_size, dropout=self.dropout
        )
        self.grn_e = GatedResidualNetwork(  # for GRN
            input_size=self.input_size, output_size=self.input_size, dropout=self.dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data with shapes (batch_size, num_timestamps, input_size)
        Returns
        -------
        :
            tuple with four context tensors with shapes (batch_size, num_timestamps, output_size)
        """
        c_s = self.grn_s(x, context=None)
        c_c = self.grn_c(x, context=None)
        c_h = self.grn_h(x, context=None)
        c_e = self.grn_e(x, context=None)
        return c_s, c_c, c_h, c_e


class TemporalFusionDecoder(nn.Module):
    """Temporal Fusion Decoder."""

    def __init__(
        self,
        input_size: int,
        n_heads: int,
        context_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Init Temporal Fusion Decoder.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        n_heads:
            number of heads in multi-head attention
        context_size:
            dimension of context
        dropout:
            dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.n_heads = n_heads
        self.context_size = context_size
        self.dropout = dropout
        self.grn1 = GatedResidualNetwork(
            input_size=self.input_size,
            output_size=self.input_size,
            dropout=self.dropout,
            context_size=self.context_size,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_size, num_heads=self.n_heads, dropout=self.dropout, batch_first=True
        )
        self.gate_norm = GateAddNorm(input_size=self.input_size, output_size=self.input_size, dropout=self.dropout)
        self.grn2 = GatedResidualNetwork(input_size=self.input_size, output_size=self.input_size, dropout=self.dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data with shapes (batch_size, num_timestamps, input_size)
        context:
            batch of data passed as the context through the block with shapes (batch_size, num_timestamps, output_size)
        Returns
        -------
        :
            output batch of data with shapes (batch_size, num_timestamps, output_size)
        """
        x = self.grn1(x, context)
        residual = x

        attn_mask = torch.triu(torch.ones(x.size()[1], x.size()[1], dtype=torch.bool), diagonal=1).to(x.device)

        x, _ = self.attention(query=x, key=x, value=x, attn_mask=attn_mask)
        x = self.gate_norm(x, residual)
        output = self.grn2(x)
        return output
