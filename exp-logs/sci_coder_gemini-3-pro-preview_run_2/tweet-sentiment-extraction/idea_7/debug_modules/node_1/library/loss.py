import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class JaccardLoss(nn.Module):
    """
    Differentiable Soft Jaccard Loss for span prediction.

    This loss approximates the word-level Jaccard score by treating the
    predicted start and end probabilities as defining a continuous span mask.

    Logic:
        P(token_i_inside) ≈ P(start <= i) * P(end >= i)
    """

    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, start_logits, end_logits, start_targets, end_targets):
        """
        Args:
            start_logits (torch.Tensor): Logits for start position (batch_size, seq_len).
            end_logits (torch.Tensor): Logits for end position (batch_size, seq_len).
            start_targets (torch.Tensor): Ground truth start indices (batch_size).
            end_targets (torch.Tensor): Ground truth end indices (batch_size).

        Returns:
            torch.Tensor: Scalar loss value (1 - mean_jaccard).
        """
        batch_size, seq_len = start_logits.size()
        device = start_logits.device

        # 1. Convert logits to probabilities
        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)

        # 2. Compute cumulative probabilities to define the span
        # P(start_index <= i)
        start_cumsum = torch.cumsum(start_probs, dim=1)

        # P(end_index >= i) = 1 - P(end_index < i)
        # We compute this efficiently using a reverse cumsum
        end_cumsum = torch.flip(
            torch.cumsum(torch.flip(end_probs, dims=[1]), dim=1), dims=[1]
        )

        # Predicted span mask: Probability that token i is inside the predicted span
        # Assumption: Independence between start and end distributions for approximation
        pred_span_mask = start_cumsum * end_cumsum

        # 3. Construct Ground Truth Span Mask
        # Create a grid of indices [0, 1, ..., seq_len-1]
        indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Expand targets for broadcasting
        start_targets_exp = start_targets.unsqueeze(1)
        end_targets_exp = end_targets.unsqueeze(1)

        # Mask is 1.0 where index is within [start_target, end_target], 0.0 otherwise
        target_span_mask = (indices >= start_targets_exp) & (indices <= end_targets_exp)
        target_span_mask = target_span_mask.float()

        # 4. Compute Soft Intersection and Union
        # Intersection = sum(pred * target)
        intersection = (pred_span_mask * target_span_mask).sum(dim=1)

        # Union = sum(pred) + sum(target) - intersection
        union = pred_span_mask.sum(dim=1) + target_span_mask.sum(dim=1) - intersection

        # 5. Compute Jaccard Score
        jaccard_score = (intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - Mean Jaccard Score
        loss = 1.0 - jaccard_score.mean()

        return loss


class TweetLoss(nn.Module):
    """
    Hybrid Loss Function combining Cross Entropy (with Label Smoothing) and Soft Jaccard Loss.

    This objective function stabilizes training via Cross Entropy while explicitly
    optimizing for the overlap metric via Jaccard Loss.
    """

    def __init__(
        self,
        label_smoothing=Config.LABEL_SMOOTHING,
        jaccard_weight=Config.JACCARD_LOSS_WEIGHT,
    ):
        """
        Args:
            label_smoothing (float): Smoothing factor for CrossEntropyLoss.
            jaccard_weight (float): Weight applied to the Jaccard loss component.
        """
        super(TweetLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.jaccard_loss_fn = JaccardLoss()
        self.jaccard_weight = jaccard_weight

    def forward(self, start_logits, end_logits, start_targets, end_targets):
        """
        Computes the weighted sum of Cross Entropy and Jaccard losses.

        Args:
            start_logits (torch.Tensor): Predicted start logits.
            end_logits (torch.Tensor): Predicted end logits.
            start_targets (torch.Tensor): Ground truth start indices.
            end_targets (torch.Tensor): Ground truth end indices.

        Returns:
            torch.Tensor: The combined loss value.
        """
        # Cross Entropy Loss (Standard classification objective)
        start_loss = self.ce(start_logits, start_targets)
        end_loss = self.ce(end_logits, end_targets)
        ce_loss = start_loss + end_loss

        # Soft Jaccard Loss (Metric-aligned objective)
        jaccard_loss = self.jaccard_loss_fn(
            start_logits, end_logits, start_targets, end_targets
        )

        # Combined Loss
        total_loss = ce_loss + (self.jaccard_weight * jaccard_loss)

        return total_loss
