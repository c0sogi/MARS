import torch
import torch.nn as nn


class AnchoredMCRMSELoss(nn.Module):
    """
    Calculates the Anchored Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function differs from the standard evaluation metric by computing the
    error over the full sequence length provided in the inputs (e.g., 0-107),
    rather than slicing to the scored length (e.g., 0-68).

    This enforces the 'Boundary Anchoring' strategy: by training on the full sequence
    where the tail targets are set to a neutral baseline (0.0), we prevent the
    hidden states of bidirectional recurrent layers from drifting in the unscored regions.
    """

    def __init__(self, scored_indices=None):
        """
        Args:
            scored_indices (list, optional): List of channel indices to include in the loss calculation.
                                             Defaults to [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C).
        """
        super(AnchoredMCRMSELoss, self).__init__()
        # Default to the competition scored columns if not provided
        if scored_indices is None:
            self.scored_indices = [0, 1, 3]
        else:
            self.scored_indices = scored_indices

    def forward(self, preds, targets):
        """
        Computes the MCRMSE loss over the full input dimensions.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Channels).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Ensure inputs are float
        preds = preds.float()
        targets = targets.float()

        column_losses = []

        for idx in self.scored_indices:
            # Extract the specific channel for all batches and all sequence positions
            # Shape: (Batch, Seq_Len)
            p = preds[:, :, idx]
            t = targets[:, :, idx]

            # Compute Mean Squared Error over the full sequence length
            # We explicitly do NOT slice the sequence here.
            mse = torch.mean((p - t) ** 2)

            # Compute Root Mean Squared Error
            # Adding a small epsilon to ensure numerical stability for gradients
            rmse = torch.sqrt(mse + 1e-8)

            column_losses.append(rmse)

        # If no columns are scored, return 0.0 with gradient tracking
        if not column_losses:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # The final loss is the average of the RMSEs across the scored columns
        loss = torch.mean(torch.stack(column_losses))

        return loss
