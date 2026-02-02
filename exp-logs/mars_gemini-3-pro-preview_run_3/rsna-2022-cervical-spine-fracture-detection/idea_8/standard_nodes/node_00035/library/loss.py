import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedMILLoss(nn.Module):
    """
    Implements the Weighted Multi-label Logarithmic Loss for Cervical Spine Fracture Detection.

    Logic:
    1. Aggregates instance-level logits via Global Max Pooling.
    2. Derives patient_overall logit as max(C1...C7).
    3. Computes Weighted BCE against global targets.

    Cite {solution_lesson_node_00033}: Removed Box-Guided supervision as it degrades performance
    due to label noise and rigid constraints.
    """

    def __init__(self):
        super().__init__()
        # Competition metric weights: C1-C7 = 1.0, Patient_Overall = 7.0
        weights = torch.tensor([1.0] * 7 + [7.0])
        self.register_buffer("mil_weights", weights)

        self.bce_mil = nn.BCEWithLogitsLoss(weight=self.mil_weights)

    def forward(self, instance_logits, targets):
        """
        Args:
            instance_logits (torch.Tensor): Shape (B, Seq_Len, 7).
            targets (torch.Tensor): Shape (B, 8).
        """
        # Global Max Pooling: (B, S, 7) -> (B, 7)
        pooled_logits, _ = torch.max(instance_logits, dim=1)

        # Derive patient_overall logit: max(C1...C7) -> (B, 1)
        patient_logit, _ = torch.max(pooled_logits, dim=1, keepdim=True)

        # Concatenate: (B, 8)
        global_logits = torch.cat([pooled_logits, patient_logit], dim=1)

        # Calculate Weighted MIL Loss
        loss = self.bce_mil(global_logits, targets.float())

        return loss, {"loss": loss.item()}
