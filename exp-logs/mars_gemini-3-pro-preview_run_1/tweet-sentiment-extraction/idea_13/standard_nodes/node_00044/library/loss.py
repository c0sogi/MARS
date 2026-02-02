import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SoftJaccardLoss(nn.Module):
    """
    Differentiable Soft Jaccard Loss.
    Optimizes the Jaccard index directly by approximating the span mask
    using cumulative sums of the start and end probabilities.
    """

    def __init__(self, epsilon=1e-7):
        super(SoftJaccardLoss, self).__init__()
        self.epsilon = epsilon

    def forward(
        self, start_logits, end_logits, start_targets, end_targets, attention_mask=None
    ):
        """
        Args:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
            start_targets: (batch_size, seq_len) - smoothed probabilities
            end_targets: (batch_size, seq_len) - smoothed probabilities
            attention_mask: (batch_size, seq_len)
        """
        # 1. Convert logits to probabilities
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # 2. Compute cumulative probabilities to approximate span membership
        # P(token_i in span) = P(start <= i) * P(end >= i)

        # P(start <= i)
        start_cumsum = torch.cumsum(start_probs, dim=-1)

        # P(end >= i) = 1 - P(end < i)
        # Calculated via reverse cumsum: flip -> cumsum -> flip
        end_cumsum = torch.flip(
            torch.cumsum(torch.flip(end_probs, dims=[-1]), dim=-1), dims=[-1]
        )

        # Predicted Span Mask (Soft)
        m_pred = start_cumsum * end_cumsum

        # 3. Compute Target Span Mask (Soft) from smoothed targets
        # We treat the smoothed targets as the ground truth distribution
        target_start_cumsum = torch.cumsum(start_targets, dim=-1)
        target_end_cumsum = torch.flip(
            torch.cumsum(torch.flip(end_targets, dims=[-1]), dim=-1), dims=[-1]
        )
        m_target = target_start_cumsum * target_end_cumsum

        # 4. Apply Attention Mask
        if attention_mask is not None:
            mask = attention_mask.type_as(m_pred)
            m_pred = m_pred * mask
            m_target = m_target * mask

        # 5. Compute Jaccard Score
        # Intersection: sum(m_pred * m_target)
        intersection = torch.sum(m_pred * m_target, dim=-1)

        # Union: sum(m_pred) + sum(m_target) - intersection
        union = torch.sum(m_pred, dim=-1) + torch.sum(m_target, dim=-1) - intersection

        jaccard = intersection / (union + self.epsilon)

        # 6. Loss is 1 - Jaccard
        loss = 1.0 - jaccard

        return loss.mean()


def compute_loss(
    start_logits, end_logits, start_targets, end_targets, attention_mask=None
):
    """
    Computes the total loss combining KL Divergence and Soft Jaccard Loss.

    Args:
        start_logits: (batch_size, seq_len)
        end_logits: (batch_size, seq_len)
        start_targets: (batch_size, seq_len) - smoothed probabilities
        end_targets: (batch_size, seq_len) - smoothed probabilities
        attention_mask: (batch_size, seq_len)

    Returns:
        total_loss: scalar tensor
    """

    # --- 1. Mask Logits for Valid Probabilities ---
    if attention_mask is not None:
        # Set logits corresponding to padding to a very small number
        # so they don't affect softmax/log_softmax
        # attention_mask is 1 for keep, 0 for remove
        # Using -10000.0 is numerically safer than -1e9 while still effectively 0 in softmax
        pad_mask = (1.0 - attention_mask.type_as(start_logits)) * -10000.0
        start_logits = start_logits + pad_mask
        end_logits = end_logits + pad_mask

    # --- 2. KL Divergence Loss ---
    # KLDivLoss expects log-probabilities as input and probabilities as target
    start_log_probs = F.log_softmax(start_logits, dim=-1)
    end_log_probs = F.log_softmax(end_logits, dim=-1)

    # reduction='batchmean' divides by batch size, which is standard
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    kl_loss_start = kl_loss_fn(start_log_probs, start_targets)
    kl_loss_end = kl_loss_fn(end_log_probs, end_targets)

    kl_loss = (kl_loss_start + kl_loss_end) / 2.0

    # --- 3. Soft Jaccard Loss ---
    # We pass the masked logits to SoftJaccardLoss
    jaccard_loss_fn = SoftJaccardLoss()
    jaccard_loss = jaccard_loss_fn(
        start_logits, end_logits, start_targets, end_targets, attention_mask
    )

    # --- 4. Combine Losses ---
    alpha = Config.LOSS_ALPHA
    total_loss = (1 - alpha) * kl_loss + alpha * jaccard_loss

    return total_loss
