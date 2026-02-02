import torch
import torch.nn as nn
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss function addresses the problem of class imbalance by down-weighting
    easy negative samples. It is a variant of Focal Loss that decouples the
    focusing parameters for positive and negative examples and introduces
    probability shifting (clipping) for negatives.

    Reference: "Asymmetric Loss For Multi-Label Classification"
    """

    def __init__(
        self,
        gamma_neg=Config.asl_gamma_neg,
        gamma_pos=Config.asl_gamma_pos,
        clip=Config.asl_clip,
        eps=1e-8,
    ):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples.
                               Higher values down-weight easy negatives more.
            gamma_pos (float): Focusing parameter for positive samples.
            clip (float): Probability margin for shifting negative samples.
                          Probabilities below this threshold for negative classes
                          will result in zero loss.
            eps (float): Small epsilon for numerical stability in logs.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits of shape (N, C).
            y (torch.Tensor): Binary targets of shape (N, C).

        Returns:
            torch.Tensor: Scalar loss value (mean reduction).
        """
        # Ensure targets are the same type as inputs (float)
        y = y.type_as(x)

        # Calculate probabilities from logits
        x_sigmoid = torch.sigmoid(x)
        p = x_sigmoid
        p_inv = 1.0 - p

        # --- Asymmetric Clipping for Negatives ---
        # If clip > 0, we shift the negative probabilities (1-p) to down-weight easy negatives.
        # Logic: If p < clip (easy negative), then 1-p > 1-clip.
        # We shift 1-p by adding clip. If it exceeds 1, we clamp it to 1.
        # When clamped to 1, log(1) = 0, so the loss for that sample becomes 0.
        if self.clip is not None and self.clip > 0:
            p_inv_shifted = (p_inv + self.clip).clamp(max=1.0)
        else:
            p_inv_shifted = p_inv

        # --- Calculate Focal Weights ---
        # Weight for positive samples: (1-p)^gamma_pos
        w_pos = torch.pow((1.0 - p).clamp(min=0.0), self.gamma_pos)

        # Weight for negative samples: (p_shifted)^gamma_neg
        # We use the shifted probability (1 - p_inv_shifted) for the weight calculation
        # to maintain consistency with the shifting logic.
        p_shifted = 1.0 - p_inv_shifted
        w_neg = torch.pow(p_shifted.clamp(min=0.0), self.gamma_neg)

        # --- Calculate Cross Entropy Terms ---
        # We use the probabilities directly in log
        log_p = torch.log(p.clamp(min=self.eps))
        log_p_inv = torch.log(p_inv_shifted.clamp(min=self.eps))

        # --- Combine Loss Components ---
        # Loss = - [ y * w_pos * log(p) + (1-y) * w_neg * log(1-p_shifted) ]
        loss_pos = -y * w_pos * log_p
        loss_neg = -(1.0 - y) * w_neg * log_p_inv

        loss = loss_pos + loss_neg

        # Return mean over the batch
        return loss.mean()
