import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HierarchicalMILLoss(nn.Module):
    """
    Implements the Hierarchical Loss for Cervical Spine Fracture Detection.
    Cite solution_lesson_node_00033: Removing box-guided supervision as it degraded performance.

    Components:
    1. L_MIL: Weighted Multi-label Logarithmic Loss on global max-pooled predictions.
       Weights: 1.0 for C1-C7, 7.0 for patient_overall.
    """

    def __init__(self):
        super().__init__()

        # Competition metric weights: C1-C7 = 1.0, Patient_Overall = 7.0
        # Registered as buffer to handle device movement automatically
        weights = torch.tensor([1.0] * 7 + [7.0])
        self.register_buffer("mil_weights", weights)

        # Base Loss Functions
        # MIL Loss: Weighted BCE. Reduction is mean to average over batch.
        self.bce_mil = nn.BCEWithLogitsLoss(weight=self.mil_weights)

    def forward(self, instance_logits, targets):
        """
        Args:
            instance_logits (torch.Tensor): Shape (B, Seq_Len, 7). Raw logits for C1-C7 per slice.
            targets (torch.Tensor): Shape (B, 8). Global ground truth [C1...C7, patient_overall].

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # --- 1. MIL Component (Global Level) ---

        # Global Max Pooling: Aggregate slice logits to study logits
        # Shape: (B, 7)
        pooled_logits, _ = torch.max(instance_logits, dim=1)

        # Derive patient_overall logit
        # Logic: If any vertebra is fractured, patient is fractured.
        # In probability space: p_overall = max(p_c1, ... p_c7)
        # In logit space: logit_overall = max(logit_c1, ... logit_c7)
        # Shape: (B, 1)
        patient_logit, _ = torch.max(pooled_logits, dim=1, keepdim=True)

        # Concatenate to form full prediction vector: [C1...C7, patient_overall]
        # Shape: (B, 8)
        global_logits = torch.cat([pooled_logits, patient_logit], dim=1)

        # Calculate Weighted MIL Loss
        loss_mil = self.bce_mil(global_logits, targets.float())

        metrics = {
            "loss": loss_mil.item(),
        }

        return loss_mil, metrics
