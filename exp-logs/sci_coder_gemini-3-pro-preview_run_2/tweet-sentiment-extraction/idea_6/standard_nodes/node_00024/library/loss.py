import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SoftJaccardLoss(nn.Module):
    """
    Soft Jaccard Loss for optimizing the Jaccard index directly.
    Computes the intersection over union between the predicted probability distribution
    and the one-hot target distribution for start and end indices.
    """

    def __init__(self):
        super(SoftJaccardLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len).
            targets (torch.Tensor): Ground truth indices of shape (batch_size).

        Returns:
            torch.Tensor: Scalar loss value (1 - Jaccard).
        """
        batch_size, seq_len = logits.size()

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Create one-hot encoded targets
        # targets: (batch_size) -> (batch_size, seq_len)
        targets_one_hot = torch.zeros(batch_size, seq_len, device=logits.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Compute Intersection: sum(probs * targets)
        # Since targets are one-hot, this effectively selects the probability of the correct class
        intersection = torch.sum(probs * targets_one_hot, dim=1)

        # Compute Union: sum(probs) + sum(targets) - intersection
        # sum(probs) is 1.0, sum(targets) is 1.0
        union = 1.0 + 1.0 - intersection

        # Jaccard Score: Intersection / Union
        # Add epsilon to avoid division by zero
        jaccard = intersection / (union + 1e-7)

        # Loss is 1 - mean Jaccard score
        loss = 1.0 - jaccard.mean()

        return loss


class HybridLoss(nn.Module):
    """
    Hybrid Loss Function combining Cross Entropy with Label Smoothing and Soft Jaccard Loss.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        # Cross Entropy with Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

        # Custom Soft Jaccard Loss
        self.jaccard_loss = SoftJaccardLoss()

        # Weighting factor
        self.jaccard_weight = Config.jaccard_weight

    def forward(self, start_logits, end_logits, start_targets, end_targets):
        """
        Args:
            start_logits (torch.Tensor): Logits for start index.
            end_logits (torch.Tensor): Logits for end index.
            start_targets (torch.Tensor): Ground truth start indices.
            end_targets (torch.Tensor): Ground truth end indices.

        Returns:
            torch.Tensor: Combined weighted loss.
        """
        # 1. Cross Entropy Loss
        start_ce = self.ce_loss(start_logits, start_targets)
        end_ce = self.ce_loss(end_logits, end_targets)
        avg_ce = (start_ce + end_ce) / 2.0

        # 2. Soft Jaccard Loss
        start_jac = self.jaccard_loss(start_logits, start_targets)
        end_jac = self.jaccard_loss(end_logits, end_targets)
        avg_jac = (start_jac + end_jac) / 2.0

        # 3. Combine Losses
        # Loss = (1 - alpha) * CE + alpha * Jaccard
        total_loss = (
            1.0 - self.jaccard_weight
        ) * avg_ce + self.jaccard_weight * avg_jac

        return total_loss
