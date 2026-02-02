import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def compute_loss(
    start_logits, end_logits, start_targets, end_targets, attention_mask=None
):
    """
    Computes the KL Divergence loss with proper masking.

    Args:
        start_logits: (batch_size, seq_len)
        end_logits: (batch_size, seq_len)
        start_targets: (batch_size, seq_len) - smoothed probabilities
        end_targets: (batch_size, seq_len) - smoothed probabilities
        attention_mask: (batch_size, seq_len)

    Returns:
        total_loss: scalar tensor
    """
    # Use KLDivLoss with reduction='none' to handle masking manually
    criterion = nn.KLDivLoss(reduction="none")

    start_log_probs = F.log_softmax(start_logits, dim=-1)
    end_log_probs = F.log_softmax(end_logits, dim=-1)

    # Compute element-wise KL divergence
    # shape: (batch_size, seq_len)
    loss_start = criterion(start_log_probs, start_targets)
    loss_end = criterion(end_log_probs, end_targets)

    if attention_mask is not None:
        mask = attention_mask.type_as(loss_start)
        loss_start = loss_start * mask
        loss_end = loss_end * mask

    # Sum over sequence length (to get KL for each sample), then mean over batch
    loss_start = loss_start.sum(dim=-1).mean()
    loss_end = loss_end.sum(dim=-1).mean()

    return (loss_start + loss_end) / 2.0
