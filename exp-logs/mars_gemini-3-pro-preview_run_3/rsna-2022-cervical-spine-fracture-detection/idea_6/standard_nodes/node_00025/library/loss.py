import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HierarchicalCompoundLoss(nn.Module):
    """
    Implements the Hierarchical Compound Loss function for cervical spine fracture detection.

    This loss function enforces logical consistency between vertebral predictions and the
    overall patient outcome. It computes:
    1. The average Binary Cross Entropy (BCE) loss for the 7 vertebral sub-types.
    2. The BCE loss for the patient_overall label, where the patient prediction is
       dynamically derived as the maximum of the vertebral predictions.

    By summing the mean vertebral loss and the patient loss, we achieve an implicit
    weighting that aligns with the competition metric (where the patient label is
    weighted 7x more than an individual vertebra).
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Computes the hierarchical loss.

        Args:
            inputs (torch.Tensor): Predicted logits for the 7 cervical vertebrae (C1-C7).
                                   Shape: (Batch_Size, 7)
            targets (torch.Tensor): Ground truth labels.
                                    Shape: (Batch_Size, 8)
                                    Expected format: [C1, C2, C3, C4, C5, C6, C7, patient_overall]

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Slice targets
        # Columns 0-6: Vertebral labels (C1-C7)
        # Column 7: Patient overall label
        target_vertebrae = targets[:, :7]
        target_patient = targets[:, 7]

        # 1. Vertebral Loss
        # Compute BCE with logits for C1-C7.
        # reduction='mean' averages the loss over the batch and the 7 classes.
        # L_vert = (1 / (B * 7)) * Sum(L_c)
        loss_vertebrae = F.binary_cross_entropy_with_logits(
            inputs, target_vertebrae, reduction="mean"
        )

        # 2. Patient Loss
        # Derive the patient prediction from the vertebral logits.
        # Logic: Patient is fractured if ANY vertebra is fractured (max probability).
        # Optimization: max(sigmoid(x)) == sigmoid(max(x)).
        # We use max(logits) directly with BCEWithLogitsLoss for better numerical stability.
        patient_logits = torch.max(inputs, dim=1).values

        # Compute BCE for the patient outcome.
        # reduction='mean' averages the loss over the batch.
        # L_patient = (1 / B) * Sum(L_p)
        loss_patient = F.binary_cross_entropy_with_logits(
            patient_logits, target_patient, reduction="mean"
        )

        # Total Loss
        # Summing these implicitly weights the patient loss higher relative to individual vertebrae.
        # Total ~ Mean(Vertebral_Losses) + Patient_Loss
        return loss_vertebrae + loss_patient
