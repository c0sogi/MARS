import torch
import torch.nn as nn


class ImplicitlyWeightedMultiTaskLoss(nn.Module):
    """
    Implements the Implicitly Weighted Multi-Task Loss.

    The loss is defined as:
    L = mean(BCE_C1...C7) + BCE_Patient

    This formulation naturally creates a 1:7 weighting ratio (relative to individual vertebrae)
    for the patient_overall label, as required by the competition metric structure,
    without using explicit scalar multipliers that might distort gradient magnitudes.
    """

    def __init__(self):
        super(ImplicitlyWeightedMultiTaskLoss, self).__init__()
        # We use BCEWithLogitsLoss which combines Sigmoid and BCE for numerical stability.
        # reduction='mean' calculates the mean over the batch and spatial/channel dimensions provided.
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
                                   Columns 0-6: C1 to C7 vertebrae.
                                   Column 7: patient_overall.
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).
                                    Same column order as logits.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float for BCEWithLogitsLoss
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)

        # Slice the inputs
        # Indices 0-6 correspond to C1, C2, C3, C4, C5, C6, C7
        vertebrae_logits = logits[:, :7]
        vertebrae_targets = targets[:, :7]

        # Index 7 corresponds to patient_overall
        patient_logits = logits[:, 7]
        patient_targets = targets[:, 7]

        # Calculate Vertebrae Loss
        # This computes the mean loss over (Batch_Size * 7) elements.
        loss_vertebrae = self.bce(vertebrae_logits, vertebrae_targets)

        # Calculate Patient Loss
        # This computes the mean loss over (Batch_Size) elements.
        loss_patient = self.bce(patient_logits, patient_targets)

        # Combine losses
        # By adding the mean of the 7 vertebrae losses to the patient loss,
        # the patient signal is effectively weighted equal to the *sum* of the 7 vertebrae signals
        # in terms of contribution to the optimization objective relative to the group.
        # Relative to a single vertebra k, the weight is 1 vs 1/7.
        total_loss = loss_vertebrae + loss_patient

        return total_loss
