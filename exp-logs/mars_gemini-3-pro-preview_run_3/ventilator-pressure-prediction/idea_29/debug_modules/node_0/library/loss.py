import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    Implements the Logic-Gated Masked L1 Loss for the Ventilator Pressure Prediction task.

    This loss function calculates the Mean Absolute Error (MAE) between predictions and targets,
    but strictly restricts the calculation to the inspiratory phase of the breath.

    The inspiratory phase is defined where the control input `u_out` is 0.
    The expiratory phase (u_out=1) is explicitly masked out and does not contribute to the gradient,
    aligning the optimization objective with the competition metric.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the masked L1 loss.

        Args:
            preds (torch.Tensor): The model predictions. Shape can be (Batch, Seq_Len) or (Batch, Seq_Len, 1).
            targets (torch.Tensor): The ground truth pressure values. Shape should match preds.
            u_out (torch.Tensor): The binary control input indicating the expiratory valve status.
                                  0 indicates inspiratory phase (valid for loss).
                                  1 indicates expiratory phase (ignored).
                                  Shape should match preds.

        Returns:
            torch.Tensor: The scalar loss value representing the MAE over the inspiratory phase.
        """
        # Flatten all tensors to 1D to ensure shape compatibility and simplify masking
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        u_out_flat = u_out.view(-1)

        # Create the boolean mask for the inspiratory phase
        # We want to keep indices where u_out is 0 (valve closed/inspiratory)
        # u_out is typically 0 or 1, but we treat it as a float mask here
        mask = (u_out_flat == 0).float()

        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds_flat - targets_flat)

        # Apply the mask: This zeros out errors occurring during the expiratory phase
        masked_diff = abs_diff * mask

        # Calculate the mean error over the valid (inspiratory) time steps
        # We divide the sum of errors by the number of valid steps (sum of the mask)
        # A small epsilon is added to the denominator to prevent division by zero in edge cases
        loss = masked_diff.sum() / (mask.sum() + 1e-8)

        return loss
