import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    References:
    "Asymmetric Loss For Multi-Label Classification", Ben-Baruch et al.
    https://arxiv.org/abs/2009.14119
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples.
            gamma_pos (float): Focusing parameter for positive samples.
            clip (float): Probability margin shift for negative samples.
            eps (float): Epsilon for numerical stability.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input logits of shape (N, C).
            y (torch.Tensor): Targets of shape (N, C). Can be binary or soft (MixUp).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping for Negatives
        # Shifts the negative probability distribution to filter out easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Calculate Focusing Weights
        # Weight for positive samples: (1 - p)^gamma_pos
        w_pos = (1 - xs_pos).pow(self.gamma_pos)
        # Weight for negative samples: (p_shifted)^gamma_neg -> which is (1 - xs_neg_shifted)^gamma_neg if thinking in terms of p
        # Here xs_neg is already the probability of being class 0 (shifted).
        # We want to penalize when p (prob of class 1) is high for a negative target.
        # If target is 0, we want p to be 0.
        # The weight is applied to the loss term log(1-p).
        # Standard Focal Loss weight for negatives is p^gamma.
        # Here p = 1 - xs_neg (roughly). So weight is (1 - xs_neg)^gamma_neg.
        w_neg = (1 - xs_neg).pow(self.gamma_neg)

        # Calculate Log Probabilities
        # Positive Log: log(p)
        # Use logsigmoid for numerical stability: log(sigmoid(x))
        log_pos = F.logsigmoid(x)

        # Negative Log: log(1 - p_shifted)
        # Since xs_neg is shifted and clamped, we calculate log directly
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        # Combine Loss Terms
        # L = - [ y * w_pos * log_pos + (1-y) * w_neg * log_neg ]
        loss = -y * w_pos * log_pos - (1 - y) * w_neg * log_neg

        # Sum over classes (dim=1), then average over batch (dim=0)
        return loss.sum(dim=1).mean()
