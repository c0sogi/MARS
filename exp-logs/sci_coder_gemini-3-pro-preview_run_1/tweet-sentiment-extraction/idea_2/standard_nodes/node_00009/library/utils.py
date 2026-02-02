import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config, seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jaccard(str1, str2):
    """
    Calculates the Jaccard similarity score between two strings.
    Metric: Intersection over Union of the set of words.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Calculates Cross Entropy Loss with Gaussian Label Smoothing.

    Instead of a hard one-hot target, this function generates a Gaussian distribution
    centered at the ground truth index. This accounts for the noise in the
    human-annotated dataset where boundaries are often fuzzy.

    Args:
        start_logits: Predicted logits for start position [batch_size, seq_len]
        end_logits: Predicted logits for end position [batch_size, seq_len]
        start_positions: Ground truth start indices [batch_size]
        end_positions: Ground truth end indices [batch_size]

    Returns:
        torch.Tensor: The scalar loss value (sum of start and end losses).
    """
    # Determine device from input logits
    device = start_logits.device
    batch_size = start_logits.size(0)
    seq_len = start_logits.size(1)

    # Sigma controls the spread of the Gaussian.
    # A value of 1.0 spreads probability to immediate neighbors, handling slight offsets.
    sigma = 1.0

    # Create a grid of indices [1, seq_len]
    grid = torch.arange(seq_len, device=device).view(1, -1).float()

    # Expand ground truth positions to [batch_size, 1] for broadcasting
    start_pos = start_positions.view(-1, 1).float()
    end_pos = end_positions.view(-1, 1).float()

    # Generate Gaussian distributions: exp(-0.5 * ((x - mu) / sigma)^2)
    start_targets = torch.exp(-0.5 * ((grid - start_pos) / sigma) ** 2)
    end_targets = torch.exp(-0.5 * ((grid - end_pos) / sigma) ** 2)

    # Normalize distributions so they sum to 1 (valid probability distributions)
    start_targets = start_targets / start_targets.sum(dim=1, keepdim=True)
    end_targets = end_targets / end_targets.sum(dim=1, keepdim=True)

    # Compute Log Softmax of model logits
    start_log_probs = F.log_softmax(start_logits, dim=1)
    end_log_probs = F.log_softmax(end_logits, dim=1)

    # Compute Soft Cross Entropy: -sum(target * log_prob)
    # We average the loss over the batch
    loss_start = -torch.sum(start_targets * start_log_probs, dim=1).mean()
    loss_end = -torch.sum(end_targets * end_log_probs, dim=1).mean()

    return loss_start + loss_end
