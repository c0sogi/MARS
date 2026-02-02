import torch
import torch.nn as nn
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Addresses class imbalance by:
    1. Asymmetric Clipping: Discarding easy negative samples (high confidence negatives).
    2. Asymmetric Focusing: Down-weighting easy samples using focal loss style weights,
       with different gammas for positive and negative samples.

    Reference: "Asymmetric Loss For Multi-Label Classification", Ben-Baruch et al.
    """

    def __init__(
        self,
        gamma_neg: float = Config.asl_gamma_neg,
        gamma_pos: float = Config.asl_gamma_pos,
        clip: float = Config.asl_clip,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples.
            gamma_pos (float): Focusing parameter for positive samples.
            clip (float): Probability shift for negative samples (margin).
            eps (float): Epsilon for numerical stability in log.
            disable_torch_grad_focal_loss (bool): If True, detaches gradients for the weighting factor.
        """
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Logits of shape (Batch, Num_Classes).
            y (torch.Tensor): Binary targets of shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Calculate probabilities from logits
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping
        # Shifts the negative probability distribution to filter out easy negatives.
        # If p_neg > 1-clip, it gets clamped to 1, resulting in 0 loss for that component.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy Calculation
        # We calculate log(p) for positives and log(p_shifted) for negatives.
        # Clamp inputs to log to avoid log(0) -> inf.
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing (Focal Loss style weighting)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Calculate pt: probability of the target class
            # For y=1, pt = xs_pos
            # For y=0, pt = xs_neg (which incorporates the clip shift)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1

            # Calculate focusing weights
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

            # Detach gradients for the weight term if requested.
            # This prevents the gradient from flowing through the modulating factor,
            # which can stabilize training.
            if self.disable_torch_grad_focal_loss:
                one_sided_w = one_sided_w.detach()

            loss *= one_sided_w

        # Return negative sum (since log is negative) averaged over batch size.
        # We sum over classes (dim 1) and mean over batch (dim 0).
        return -loss.sum() / x.size(0)
