import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples. Higher values down-weight easy negatives more.
            gamma_pos (float): Focusing parameter for positive samples.
            clip (float): Probability margin for negative samples. Predictions below this are treated as 0 error.
            eps (float): Epsilon for numerical stability in log.
            disable_torch_grad_focal_loss (bool): If True, detaches gradients for the weighting factor.
        """
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input logits of shape (N, C).
            y (torch.Tensor): Targets of shape (N, C).

        Returns:
            torch.Tensor: Scalar loss value (Mean over batch, Sum over classes).
        """

        # Ensure targets are float
        y = y.float()

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping
        # For negative samples, we shift the probability of being negative (xs_neg) upwards
        # effectively shifting the probability of being positive (p) downwards.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy Terms
        # Loss for positive examples: L+ = - y * (1 - p)^gamma_pos * log(p)
        # Loss for negative examples: L- = - (1 - y) * (p_shifted)^gamma_neg * log(1 - p_shifted)
        # Note: (1 - p_shifted) corresponds to our clipped xs_neg.

        # Weighting Factors
        # w_pos = (1 - p)^gamma_pos
        # w_neg = (1 - xs_neg_shifted)^gamma_neg
        w_pos = (1.0 - xs_pos).pow(self.gamma_pos)
        w_neg = (1.0 - xs_neg).pow(self.gamma_neg)

        if self.disable_torch_grad_focal_loss:
            w_pos = w_pos.detach()
            w_neg = w_neg.detach()

        # Log probabilities
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        # Final Loss Calculation
        # We sum over classes (dim=1) to treat each sample's label set as a whole,
        # then mean over the batch (dim=0) to be batch-size independent.
        loss = -(y * w_pos * log_pos + (1 - y) * w_neg * log_neg)

        return loss.sum() / x.size(0)
