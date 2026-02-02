import torch
import torch.nn as nn


class ImplicitWeightedLoss(nn.Module):
    """
    Implicitly Weighted Multi-Task Loss.

    Implements the loss formulation: L = mean(BCE_C1...C7) + BCE_Patient

    This strategy naturally enforces a higher importance on the 'patient_overall'
    label compared to individual vertebrae labels. By averaging the loss across
    the 7 vertebrae columns, the gradient contribution of the vertebrae group
    is balanced against the single patient head, effectively creating a 1:7
    weighting ratio for individual vertebrae versus the patient outcome.
    """

    def __init__(self):
        super().__init__()
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        # reduction='mean' ensures we calculate the average loss over the batch (and classes).
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted multi-task loss.

        Args:
            logits (torch.Tensor): Predicted logits from the model of shape (Batch, 8).
                                   Columns 0-6 correspond to C1-C7.
                                   Column 7 corresponds to patient_overall.
            targets (torch.Tensor): Ground truth binary labels of shape (Batch, 8).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Separate vertebrae predictions/targets (first 7 columns)
        # and patient prediction/target (last column, index 7)
        c_logits = logits[:, :7]
        c_targets = targets[:, :7]

        p_logits = logits[:, 7]
        p_targets = targets[:, 7]

        # Calculate Mean BCE for C1-C7
        # This averages the loss over the batch size AND the 7 vertebrae classes.
        loss_vertebrae = self.bce(c_logits, c_targets)

        # Calculate BCE for Patient Overall
        # This averages the loss over the batch size.
        loss_patient = self.bce(p_logits, p_targets)

        # Total Loss = Mean(Vertebrae Loss) + Patient Loss
        total_loss = loss_vertebrae + loss_patient

        return total_loss
