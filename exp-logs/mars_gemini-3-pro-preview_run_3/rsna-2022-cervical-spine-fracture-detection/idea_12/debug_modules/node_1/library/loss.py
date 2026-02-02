import torch
import torch.nn as nn


class ImplicitWeightedLoss(nn.Module):
    """
    Implicitly Weighted Multi-Task Loss.

    Implements the formulation: L = mean(BCE_C1..C7) + BCE_Patient.

    This loss function calculates Binary Cross Entropy with Logits for all targets.
    It then aggregates the losses such that the 'patient_overall' label contributes
    significantly to the gradient (equivalent to the sum of average vertebrae losses),
    aligning with the competition's weighted metric where the patient-level outcome
    is weighted more highly.

    Expected Input Shapes:
        logits: (Batch_Size, 8) -> [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        targets: (Batch_Size, 8) -> [C1, C2, C3, C4, C5, C6, C7, patient_overall]
    """

    def __init__(self):
        super(ImplicitWeightedLoss, self).__init__()
        # Use reduction='none' to obtain element-wise losses for custom aggregation
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits from the model. Shape (B, 8).
            targets (torch.Tensor): Ground truth binary labels. Shape (B, 8).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute element-wise BCE loss
        # Shape: (Batch_Size, 8)
        all_losses = self.bce(logits, targets)

        # Separate vertebrae losses (indices 0-6) and patient loss (index 7)
        # vertebrae_losses shape: (Batch_Size, 7)
        vertebrae_losses = all_losses[:, :7]

        # patient_losses shape: (Batch_Size, )
        patient_losses = all_losses[:, 7]

        # Calculate the mean loss for vertebrae (averaged over batch and the 7 classes)
        # This implicitly down-weights individual vertebrae relative to the patient label
        v_loss_mean = vertebrae_losses.mean()

        # Calculate the mean loss for patient overall (averaged over batch)
        p_loss_mean = patient_losses.mean()

        # Total loss is the sum of the two components
        total_loss = v_loss_mean + p_loss_mean

        return total_loss
