import torch
import torch.nn as nn


class RSNALoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss for RSNA Cervical Spine Fracture Detection.

    Metric:
        L = Mean( w_j * BCE_{ij} ) over all i (samples) and j (rows/labels).
        Weights (w_j):
            - C1-C7: 1.0
            - Patient Overall: 7.0 (Weighted more highly as per task description)

    This matches the competition metric where loss is averaged across all 8 rows per exam.
    """

    def __init__(self):
        super().__init__()
        # Use reduction='none' to apply element-wise weights before averaging
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Define weights: 50% for patient, 50% for vertebrae (split 7 ways)
        # Cite Lesson 72: Unit-sum weights for metric stability
        w_patient = 0.5
        w_vert = 0.5 / 7
        weights = [w_vert] * 7 + [w_patient]
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).

        Returns:
            torch.Tensor: Scalar loss value (Weighted Log Loss).
        """
        # Calculate BCE for each element (Batch, 8)
        loss = self.bce(logits, targets)

        # Apply weights element-wise (Broadcasting)
        weighted_loss = loss * self.weights

        # Sum across columns (classes) to get per-sample weighted loss, then mean across batch
        return weighted_loss.sum(dim=1).mean()
