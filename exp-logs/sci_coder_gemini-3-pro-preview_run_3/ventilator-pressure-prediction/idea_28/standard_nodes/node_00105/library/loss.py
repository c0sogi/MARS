import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    Logic-Gated Masked L1 Loss.

    Calculates the Mean Absolute Error (MAE) between predictions and targets,
    but strictly enforces the metric definition by masking out the expiratory phase.

    The loss is calculated only where u_out == 0 (inspiratory phase).
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        # We use reduction='none' to get element-wise loss, which we then mask manually
        self.l1 = nn.L1Loss(reduction="none")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the masked L1 loss.

        Args:
            pred (torch.Tensor): The predicted pressure values. Shape: (Batch, ...)
            target (torch.Tensor): The ground truth pressure values. Shape: (Batch, ...)
            u_out (torch.Tensor): The expiratory valve control input (0 or 1).
                                  Shape must be broadcastable to pred/target.
                                  0 indicates inspiratory phase (included in loss).
                                  1 indicates expiratory phase (excluded from loss).

        Returns:
            torch.Tensor: The scalar loss value averaged over the inspiratory phase samples.
        """
        # Ensure inputs are on the same device (usually handled by caller, but good for safety)
        # Calculate element-wise absolute error
        loss = self.l1(pred, target)

        # Create the mask.
        # u_out is 1 for expiration, 0 for inspiration.
        # We want to keep indices where u_out is 0.
        # Depending on input processing, u_out might be float or int.
        # We assume u_out contains 0s and 1s.
        mask = 1.0 - u_out

        # Ensure mask is the same shape as loss (handle potential singleton dimensions)
        if mask.shape != loss.shape:
            mask = mask.view_as(loss)

        # Apply the mask to the loss
        masked_loss = loss * mask

        # Calculate the mean over the valid (inspiratory) elements
        # Sum of mask gives the number of valid elements
        # Add a small epsilon to avoid division by zero in case a batch has no inspiratory phase (unlikely)
        loss_val = masked_loss.sum() / (mask.sum() + 1e-8)

        return loss_val
