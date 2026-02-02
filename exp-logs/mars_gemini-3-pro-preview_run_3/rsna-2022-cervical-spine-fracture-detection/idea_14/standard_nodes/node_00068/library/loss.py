import torch
import torch.nn as nn


class RSNALoss(nn.Module):
    """
    Implicitly Weighted Multi-Task Loss for RSNA Cervical Spine Fracture Detection.

    Objective:
        L = mean(BCE_C1..C7) + BCE_Patient

    This formulation balances the detailed vertebrae-level detection (averaged)
    with the high-priority patient-level diagnosis (single), effectively giving
    higher relative weight to the patient outcome as required by the metric logic.
    """

    def __init__(self):
        super().__init__()
        # We use reduction='mean' to compute the average loss over the batch
        # and spatial/class dimensions automatically.
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
                                   Columns 0-6: C1-C7
                                   Column 7: patient_overall
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Separate Vertebrae (C1-C7) and Patient Overall components
        # Logits shape: (B, 8) -> Slicing gives (B, 7) and (B)

        # 1. Vertebrae Loss: Mean BCE over C1-C7
        # reduction='mean' divides by (Batch * 7)
        vert_logits = logits[:, :7]
        vert_targets = targets[:, :7]
        loss_vert = self.bce(vert_logits, vert_targets)

        # 2. Patient Overall Loss: BCE for the patient label
        # reduction='mean' divides by Batch
        pat_logits = logits[:, 7]
        pat_targets = targets[:, 7]
        loss_pat = self.bce(pat_logits, pat_targets)

        # Total Loss = Mean(Vertebrae) + Patient
        return loss_vert + loss_pat
