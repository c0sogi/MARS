import torch
import torch.nn as nn


class HierarchicalCompoundLoss(nn.Module):
    """
    Hierarchical Compound Loss for Cervical Spine Fracture Detection.

    This loss function enforces logical consistency between vertebral predictions
    and the patient-level outcome. It computes Binary Cross Entropy (BCE) for
    the 7 vertebral sub-labels and a separate BCE for the patient_overall label.

    Weighting Strategy:
    - Vertebral Loss: Mean over (Batch * 7) elements.
    - Patient Loss: Mean over (Batch) elements.
    - Total Loss = Vertebral Loss + Patient Loss.

    This implicit weighting ensures that the patient-level outcome (which is the
    logical OR of the vertebrae) is weighted heavily enough to drive calibration,
    while the specific vertebrae labels provide detailed supervision.
    """

    def __init__(self):
        super(HierarchicalCompoundLoss, self).__init__()
        # We use BCEWithLogitsLoss because the model outputs logits.
        # reduction='mean' ensures we get the average loss per element in the slice.
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
                                   Columns: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).
                                    Columns: [C1, C2, C3, C4, C5, C6, C7, patient_overall]

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Vertebral Loss (Columns 0-6)
        # Slicing gives shape (Batch, 7)
        vertebrae_logits = logits[:, :7]
        vertebrae_targets = targets[:, :7]

        # Calculate mean BCE over all vertebral predictions
        loss_vertebrae = self.bce(vertebrae_logits, vertebrae_targets)

        # 2. Patient Overall Loss (Column 7)
        # Slicing with range 7:8 keeps the dimension -> shape (Batch, 1)
        patient_logits = logits[:, 7:8]
        patient_targets = targets[:, 7:8]

        # Calculate mean BCE over patient predictions
        loss_patient = self.bce(patient_logits, patient_targets)

        # 3. Compound Loss
        # Summing them balances the specific supervision with the global outcome
        total_loss = loss_vertebrae + loss_patient

        return total_loss
