import torch
import torch.nn as nn
from library.config import Config


class HierarchicalCompoundLoss(nn.Module):
    """
    Hierarchical Compound Loss for Cervical Spine Fracture Detection.

    Computes a weighted sum of:
    1. Mean Binary Cross Entropy for specific vertebrae (C1-C7).
    2. Binary Cross Entropy for the 'patient_overall' outcome, where the
       prediction is derived as the maximum of the vertebrae logits.

    This enforces logical consistency: if the model predicts a high probability
    for any vertebra, the patient probability must also be high.
    """

    def __init__(self):
        super(HierarchicalCompoundLoss, self).__init__()
        # Use reduction='mean' to average losses over the batch
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits for C1-C7. Shape (Batch, 7).
            targets (torch.Tensor): Ground truth labels. Shape (Batch, 8).
                                    Columns 0-6 correspond to C1-C7.
                                    Column 7 corresponds to patient_overall.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # --- 1. Vertebrae Loss (C1-C7) ---
        # Extract targets for C1-C7 (first 7 columns)
        # Config.NUM_CLASSES is 7
        c1_c7_targets = targets[:, : Config.NUM_CLASSES]

        # Calculate mean BCE loss across all vertebrae predictions
        loss_vertebrae = self.bce(logits, c1_c7_targets)

        # --- 2. Patient Overall Loss ---
        # Derive patient_overall logit as the maximum of C1-C7 logits.
        # Logic: If any vertebra is fractured (high logit), patient is fractured.
        # Shape: (Batch, 7) -> (Batch)
        patient_derived_logits, _ = torch.max(logits, dim=1)

        # Extract target for patient_overall (8th column, index 7)
        patient_targets = targets[:, Config.NUM_CLASSES]

        # Calculate BCE loss for the derived patient prediction
        loss_patient = self.bce(patient_derived_logits, patient_targets)

        # --- 3. Total Loss ---
        # Implicit Weighting: Summing the mean of subtypes and the patient loss.
        # This effectively weights the patient_overall prediction equal to the
        # aggregate of all subtypes, satisfying the requirement to weight the
        # 'any' label more highly than specific subtypes.
        total_loss = loss_vertebrae + loss_patient

        return total_loss
