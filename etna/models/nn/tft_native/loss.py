from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    from torch.nn.modules.loss import _Loss


class QuantileLoss(_Loss):
    """Loss."""

    @staticmethod
    def forward(inputs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs:
            true target values
        pred:
            pred values.

        Returns
        -------
        :
            loss

        """
        return inputs + pred
