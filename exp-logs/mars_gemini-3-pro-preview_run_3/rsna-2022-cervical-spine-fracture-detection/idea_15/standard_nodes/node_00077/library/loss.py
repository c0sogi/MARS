import torch
import torch.nn as nn


class ImplicitWeightedLoss(nn.Module):
    """
    Implements the Implicitly Weighted Multi-Task Loss.

    Strategy:
        L = mean(BCE_C1..C7) + BCE_Patient

    This formulation naturally creates a 1:7 weighting ratio between the patient outcome
    and individual vertebrae, aligning with the competition metric which weights the
    patient-level label higher, without requiring explicit scalar multipliers.
    """

    def __init__(self):
        super().__init__()
        # We use reduction='none' to obtain the loss for every element (Batch, 8)
        # so we can manually apply the specific aggregation logic.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
                                   Columns 0-6: C1-C7
                                   Column 7: patient_overall
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).

        Returns:
            torch.Tensor: Scalar loss value averaged over the batch.
        """
        # Compute binary cross entropy for all targets
        # loss_matrix shape: (Batch, 8)
        loss_matrix = self.bce(logits, targets)

        # Separate the losses
        # c_losses shape: (Batch, 7) - Losses for C1 through C7
        c_losses = loss_matrix[:, :7]

        # p_losses shape: (Batch,) - Loss for patient_overall (column index 7)
        p_losses = loss_matrix[:, 7]

        # Calculate the composite loss per sample
        # We take the mean of the vertebral losses and add the patient loss.
        # This effectively weights the patient loss 7 times higher than any single vertebral loss.
        # Cite solution_lesson_node_00043: Normalize by sum of weights (2) to match weighted average metric.
        per_sample_loss = (c_losses.mean(dim=1) + p_losses) / 2.0

        # Return the mean loss over the batch
        return per_sample_loss.mean()
