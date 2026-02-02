import torch
import torch.nn as nn
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss function addresses two main issues in multi-label classification:
    1. Class Imbalance: By down-weighting easy negatives (which are the majority).
    2. Label Noise: By using a margin (clip) to ignore small probabilities for negatives.

    Reference: "Asymmetric Loss For Multi-Label Classification", Ben-Baruch et al.
    """

    def __init__(
        self,
        gamma_neg=Config.asl_gamma_neg,
        gamma_pos=Config.asl_gamma_pos,
        clip=Config.asl_clip,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits from the model of shape (N, C).
            y (torch.Tensor): Binary targets of shape (N, C).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        # For negatives, we shift the probability: p_neg_shifted = max(p_neg - clip, 0)
        # In terms of (1-p), this means we increase the value towards 1.
        # xs_neg stores (1-p). We want to shift p down, which means shifting (1-p) up.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross-Entropy Calculation
        # L+ = - y * log(p)
        # L- = - (1-y) * log(p_neg_shifted)
        # Note: xs_neg is the probability of the "negative class" (1-p) after shifting.
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        # Weights: w+ = (1-p)^gamma_pos, w- = (p_shifted)^gamma_neg
        # Note: p_shifted = 1 - xs_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # Calculate pt: probability of the target class
            # For y=1, pt = p (xs_pos)
            # For y=0, pt = 1 - p_shifted = xs_neg

            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1

            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss *= one_sided_w

        # Sum over classes, mean over batch
        # This reduction strategy is standard for ASL to maintain gradient magnitude
        return -loss.sum() / x.size(0)
