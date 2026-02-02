import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import set_seed


class RDropLoss(nn.Module):
    """
    Implements R-Drop (Regularized Dropout) Loss for multi-label classification.

    This loss minimizes the Binary Cross Entropy for two stochastic forward passes
    of the same input, while enforcing consistency between their outputs using
    symmetric Kullback-Leibler divergence.

    Formula:
        L_total = BCE(pred1, target) + BCE(pred2, target) + alpha * KL(pred1 || pred2)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha (float): Weighting coefficient for the KL divergence consistency term.
        """
        super(RDropLoss, self).__init__()
        self.alpha = alpha

    def forward(
        self, logits1: torch.Tensor, logits2: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the R-Drop loss.

        Args:
            logits1 (torch.Tensor): Logits from the first forward pass. Shape (Batch, NumLabels).
            logits2 (torch.Tensor): Logits from the second forward pass. Shape (Batch, NumLabels).
            targets (torch.Tensor): Ground truth targets in range [0, 1]. Shape (Batch, NumLabels).

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # 1. Compute Binary Cross Entropy Loss for both forward passes
        # binary_cross_entropy_with_logits is numerically stable and handles continuous targets [0,1]
        loss_bce1 = F.binary_cross_entropy_with_logits(logits1, targets)
        loss_bce2 = F.binary_cross_entropy_with_logits(logits2, targets)

        # 2. Compute Symmetric KL Divergence for Consistency
        # Since we have multi-label classification, each label is an independent Bernoulli distribution.
        # We must compute KL between the distributions [p, 1-p] for each label.

        # Probabilities
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)

        # Log-Probabilities (using logsigmoid for stability)
        # Construct (Batch, NumLabels, 2) tensors representing P(y=1) and P(y=0)
        log_p1 = torch.stack([F.logsigmoid(logits1), F.logsigmoid(-logits1)], dim=-1)
        log_p2 = torch.stack([F.logsigmoid(logits2), F.logsigmoid(-logits2)], dim=-1)

        # Full probability distributions for the target argument in F.kl_div
        p1_dist = torch.stack([p1, 1.0 - p1], dim=-1)
        p2_dist = torch.stack([p2, 1.0 - p2], dim=-1)

        # F.kl_div(input, target) computes target * (log(target) - input)
        # KL(p1 || p2) -> input=log_p2, target=p1
        # reduction='none' gives shape (Batch, NumLabels, 2)
        kl_1 = F.kl_div(log_p2, p1_dist, reduction="none").sum(
            dim=-1
        )  # Sum over the 2 states (Bernoulli)

        # KL(p2 || p1) -> input=log_p1, target=p2
        kl_2 = F.kl_div(log_p1, p2_dist, reduction="none").sum(dim=-1)

        # Average over batch and labels to match the scale of BCE (which defaults to mean reduction)
        kl_loss = 0.5 * (kl_1.mean() + kl_2.mean())

        # 3. Total Loss
        total_loss = loss_bce1 + loss_bce2 + self.alpha * kl_loss

        return total_loss
