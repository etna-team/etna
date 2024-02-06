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

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1) -> None:
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

    def forward(self, x: torch.tensor) -> torch.tensor:
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
        x = torch.matmul(a, b)
        return x


class GateAddNorm(nn.Module):
    """Gated Add&Norm layer."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1) -> None:
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

    def forward(self, x: torch.tensor, residual: torch.tensor) -> torch.tensor:
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

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1, context: bool = False) -> None:
        """Init GRN.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        output_size:
            output size of the feature representation
        dropout:
            dropout rate
        context:
            whether to pass context vector through the block
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.context = context

        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.elu = nn.ELU()

        if self.context:
            self.context_fc = nn.Linear(self.input_size, self.input_size, bias=False)

        self.residual_fc = nn.Linear(self.input_size, self.output_size) if self.input_size != self.output_size else None
        self.fc2 = nn.Linear(self.input_size, self.input_size)

        self.gate_norm = GateAddNorm(
            input_size=self.input_size,
            output_size=self.output_size,
            dropout=self.dropout,
        )

    def forward(self, x: torch.tensor, context: Optional[torch.tensor] = None) -> torch.tensor:
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

    def __init__(self, input_size: int, features: List = (), context: bool = False, dropout: float = 0.1) -> None:
        """Init Variable Selection Network.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        features:
            features to pass through the block
        context:
            whether to pass context vector through the block
        dropout:
            dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.features = features
        self.context = context
        self.dropout = dropout
        self.grns = {
            feature: GatedResidualNetwork(
                input_size=self.input_size,
                output_size=self.input_size,
                dropout=self.dropout,
                context=False,
            )
            for feature in self.features
        }
        self.flatten_grn = GatedResidualNetwork(
            input_size=self.input_size * self.num_features,
            output_size=self.input_size,
            dropout=self.dropout,
            context=self.context,
        )
        self.softmax = nn.Softmax()

    @property
    def num_features(self) -> int:
        """Get number of all features.

        Returns
        -------
        :
            number of all features.
        """
        return len(self.features)

    def forward(self, x: Dict[str : torch.tensor], context: Optional[torch.tensor] = None) -> torch.tensor:
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
        output = []  # TODO try 4-dimention tensor
        for feature, embedding in x.items():
            output.append(self.grns[feature](embedding))
        output = torch.stack(output, dim=-1)
        flatten_input = torch.cat(list(x.values()), dim=-1)
        flatten_grn_output = self.flatten_grn(x=flatten_input, context=context)
        feature_weights = self.softmax(flatten_grn_output).unsqueeze(dim=-2)
        output = (output * feature_weights).sum(dim=-1)
        return output


class StaticCovariateEncoder(nn.Module):
    """Static Covariate Encoder."""

    def __init__(self, input_size: int, output_size: int, output_flatten_size: int, dropout: float = 0.1) -> None:
        """Init Static Covariate Encoder.

        Parameters
        ----------
        input_size:
            input size of the feature representation
        output_size:
            output size of the feature representation
        output_flatten_size:
            output size of the VariableSelectionNetwork context vector
        dropout:
            dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_flatten_size = output_flatten_size
        self.dropout = dropout
        self.grn_s = GatedResidualNetwork(  # for VariableSelectionNetwork
            input_size=self.input_size, output_size=self.output_flatten_size, dropout=self.dropout, context=False
        )
        self.grn_c = GatedResidualNetwork(  # for LSTM
            input_size=self.input_size, output_size=self.output_size, dropout=self.dropout, context=False
        )
        self.grn_h = GatedResidualNetwork(  # for LSTM
            input_size=self.input_size, output_size=self.output_size, dropout=self.dropout, context=False
        )
        self.grn_e = GatedResidualNetwork(  # for GRN
            input_size=self.input_size, output_size=self.output_size, dropout=self.dropout, context=False
        )

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
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
