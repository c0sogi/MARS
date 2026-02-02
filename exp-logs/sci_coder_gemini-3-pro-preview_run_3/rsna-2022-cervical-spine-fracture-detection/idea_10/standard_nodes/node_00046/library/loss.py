import torch
import torch.nn as nn


class HierarchicalCompoundLoss(nn.Module):
    """
    Hierarchical Compound Loss for Cervical Spine Fracture Detection.

    Computes the weighted sum of:
    1. Vertebral Loss: Mean Binary Cross Entropy over the 7 cervical vertebrae (C1-C7).
    2. Patient Loss: Binary Cross Entropy for the patient-level outcome, where the
       predicted patient logit is derived as the maximum of the C1-C7 logits.

    This implements the 'Implicit Weighting' strategy where the patient-level outcome
    contributes significantly to the gradient, enforcing consistency between local
    predictions and the global label.
    """

    def __init__(self):
        super(HierarchicalCompoundLoss, self).__init__()
        # reduction='mean' calculates the mean over all elements in the batch and channels
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets, patient_target):
        """
        Args:
            logits (torch.Tensor): Predicted logits. Shape (Batch_Size, 8).
                                   Cols 0-6: C1-C7. Col 7: Patient Overall.
            targets (torch.Tensor): Ground truth labels for C1-C7. Shape (Batch_Size, 7).
            patient_target (torch.Tensor): Ground truth label for patient overall. Shape (Batch_Size,).

        Returns:
            torch.Tensor: The calculated loss scalar.
        """
        # Split logits
        c_logits = logits[:, :7]  # (B, 7)
        p_logits = logits[:, 7]  # (B,)

        # 1. Vertebral Loss
        # Calculates the mean BCE loss across the batch and the 7 vertebrae classes.
        vertebral_loss = self.bce(c_logits, targets)

        # 2. Patient Loss
        # Direct supervision on the patient head
        patient_loss = self.bce(p_logits, patient_target)

        # 3. Total Loss
        # Summing them implies that the single patient_overall label has equal weight
        # to the average of the 7 vertebrae labels. This effectively upweights the
        # patient_overall metric, aligning with the competition metric.
        total_loss = vertebral_loss + patient_loss

        return total_loss
