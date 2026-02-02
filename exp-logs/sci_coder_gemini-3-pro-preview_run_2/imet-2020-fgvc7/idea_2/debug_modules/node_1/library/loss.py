import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss function addresses the problem of extreme class imbalance in multi-label
    datasets by decoupling the focusing parameters for positive and negative samples.
    It specifically down-weights 'easy negatives'—negative samples that the model
    classifies correctly with high confidence—preventing them from dominating the
    gradient.

    Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    """

    def __init__(
        self,
        gamma_neg=Config.ASL_GAMMA_NEG,
        gamma_pos=Config.ASL_GAMMA_POS,
        clip=Config.ASL_CLIP,
        eps=1e-8,
    ):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples. Higher values
                               down-weight easy negatives more aggressively.
            gamma_pos (float): Focusing parameter for positive samples.
            clip (float): Probability margin for shifting negative samples.
                          Probabilities below this value are zeroed out for the negative loss.
            eps (float): Small constant for numerical stability.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Forward pass of the loss function.

        Args:
            x (torch.Tensor): Input logits from the model of shape (N, C).
            y (torch.Tensor): Target multi-hot encoded labels of shape (N, C).

        Returns:
            torch.Tensor: Scalar loss value (averaged over batch, summed over classes).
        """
        # Calculate probabilities from logits
        p = torch.sigmoid(x)

        # --- Positive Samples (Target = 1) ---
        # Standard Focal Loss logic for positives: - (1-p)^gamma * log(p)
        # We use F.logsigmoid(x) which is equivalent to log(sigmoid(x)) but more stable

        if self.gamma_pos > 0:
            # (1 - p) can be computed as sigmoid(-x)
            pos_weight = (1 - p) ** self.gamma_pos
        else:
            pos_weight = 1.0

        # Loss contribution from positive samples
        # y acts as a mask
        loss_pos = -y * pos_weight * F.logsigmoid(x)

        # --- Negative Samples (Target = 0) ---
        # Asymmetric logic: Shift probability p by clip margin
        # p_shifted = max(p - clip, 0)

        if self.clip > 0:
            p_shifted = F.relu(p - self.clip)
        else:
            p_shifted = p

        # Weighting factor for negatives
        neg_weight = p_shifted**self.gamma_neg

        # Loss contribution from negative samples
        # (1-y) acts as a mask
        # We compute log(1 - p_shifted). Added eps to prevent log(0).
        loss_neg = -(1 - y) * neg_weight * torch.log(1 - p_shifted + self.eps)

        # --- Total Loss ---
        # Sum losses over all classes, then average over the batch
        loss = loss_pos + loss_neg
        return loss.sum() / x.size(0)
