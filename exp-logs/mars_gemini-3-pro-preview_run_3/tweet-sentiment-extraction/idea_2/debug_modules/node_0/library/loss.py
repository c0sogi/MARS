import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TweetLoss(nn.Module):
    """
    Compound loss function for Tweet Sentiment Extraction.
    Combines CrossEntropyLoss for start/end indices with a differentiable Soft Jaccard Loss
    to directly optimize the competition metric.
    """

    def __init__(self, soft_jaccard_weight=None):
        super(TweetLoss, self).__init__()
        self.soft_jaccard_weight = (
            soft_jaccard_weight
            if soft_jaccard_weight is not None
            else Config.SOFT_JACCARD_WEIGHT
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        """
        Calculates the compound loss.

        Args:
            start_logits (torch.Tensor): Predicted logits for start index (Batch, Seq_Len).
            end_logits (torch.Tensor): Predicted logits for end index (Batch, Seq_Len).
            start_positions (torch.Tensor): Ground truth start indices (Batch).
            end_positions (torch.Tensor): Ground truth end indices (Batch).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # 1. Standard Cross Entropy Loss
        # We average the loss for start and end positions
        start_ce = self.ce_loss(start_logits, start_positions)
        end_ce = self.ce_loss(end_logits, end_positions)
        ce_loss = (start_ce + end_ce) / 2

        # If weight is 0, return only CE loss to save computation
        if self.soft_jaccard_weight == 0:
            return ce_loss

        # 2. Soft Jaccard Loss
        # Convert logits to probabilities
        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)

        # Generate Soft Predicted Mask
        # The probability that token 'i' is inside the span is:
        # P(i in span) = P(start <= i) * P(end >= i)
        # Assuming independence between start and end distributions.

        # P(start <= i) is the cumulative sum of start probabilities from 0 to i
        pred_start_cum = torch.cumsum(start_probs, dim=1)

        # P(end >= i) is the cumulative sum of end probabilities from i to L-1
        # We compute this by flipping, cumsum, and flipping back
        pred_end_cum = torch.cumsum(end_probs.flip(1), dim=1).flip(1)

        # Soft mask: (Batch, Seq_Len)
        pred_mask = pred_start_cum * pred_end_cum

        # Generate Ground Truth Mask
        batch_size, seq_len = start_logits.size()
        device = start_logits.device

        # Create grid of indices: [0, 1, ..., seq_len-1] repeated for batch
        indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Mask is 1 where start_pos <= index <= end_pos
        # Use unsqueeze to broadcast positions against indices
        target_mask = (indices >= start_positions.unsqueeze(1)) & (
            indices <= end_positions.unsqueeze(1)
        )
        target_mask = target_mask.float()

        # Compute Jaccard Score
        # Intersection = sum(pred * target)
        intersection = (pred_mask * target_mask).sum(dim=1)

        # Union = sum(pred) + sum(target) - intersection
        union = pred_mask.sum(dim=1) + target_mask.sum(dim=1) - intersection

        # Add epsilon to prevent division by zero
        epsilon = 1e-7
        jaccard_score = (intersection + epsilon) / (union + epsilon)

        # Loss is 1 - Mean Jaccard Score
        jaccard_loss = 1.0 - jaccard_score.mean()

        # 3. Combine Losses
        total_loss = ce_loss + (self.soft_jaccard_weight * jaccard_loss)

        return total_loss
