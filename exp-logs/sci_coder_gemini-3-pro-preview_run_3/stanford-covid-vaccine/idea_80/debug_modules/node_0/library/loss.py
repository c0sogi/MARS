import torch
import torch.nn as nn
from library.utils import mcrmse


class MCRMSELoss(nn.Module):
    """
    MCRMSELoss implements the Mean Columnwise Root Mean Squared Error loss function.

    This loss is used to optimize the model on all 5 target columns:
    'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'.

    It flattens the input tensors to ensure the metric is calculated globally
    across the batch and sequence length for each target column.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values.
                                   Shape can be (Batch, Num_Targets) or (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values.
                                    Shape must match inputs.

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # Determine the number of target columns (last dimension)
        num_targets = inputs.shape[-1]

        # Flatten the batch and sequence dimensions to (N, Num_Targets)
        # This ensures that the MSE is calculated over all samples and positions
        # for each specific target column.
        inputs_flat = inputs.view(-1, num_targets)
        targets_flat = targets.view(-1, num_targets)

        # Use the provided utility function to calculate the metric
        return mcrmse(targets_flat, inputs_flat)
