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

        # Define weights: 1/7 for each vertebra, 1 for patient_overall
        # Cite solution_lesson_node_00043: Absolute weight scaling dictates metric comparability.
        self.register_buffer(
            "weights",
            torch.tensor(
                [1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0]
            ),
        )

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

        # Average across all rows (Batch * 8 elements)
        return weighted_loss.mean()
