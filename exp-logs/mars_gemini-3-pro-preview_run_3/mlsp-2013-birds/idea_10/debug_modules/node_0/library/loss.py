import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss function addresses the problem of class imbalance by dynamically
    down-weighting easy negative samples. It is defined as:
    L = -y * L_pos - (1-y) * L_neg

    where:
    L_pos = (1 - p)^gamma_pos * log(p)
    L_neg = (p_m)^gamma_neg * log(1 - p_m)
    p_m = max(p - clip, 0)

    Args:
        gamma_neg (float): Focusing parameter for negative samples. Higher values down-weight easy negatives more.
        gamma_pos (float): Focusing parameter for positive samples.
        clip (float): Margin shift for negative samples. Probabilities below this threshold are zeroed out.
        eps (float): Small epsilon for numerical stability.
        disable_torch_grad_focal_loss (bool): If True, detaches weights from the computation graph (standard practice).
    """

    def __init__(
        self,
        gamma_neg=4.0,
        gamma_pos=1.0,
        clip=0.05,
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
        Forward pass.

        Args:
            x (torch.Tensor): Logits from the model of shape (batch_size, num_classes).
            y (torch.Tensor): Binary targets of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Calculate probabilities from logits
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid_pos = x_sigmoid
        x_sigmoid_neg = 1 - x_sigmoid

        # --- Positive Part ---
        # L+ = (1-p)^gamma_pos * log(p)
        # We use log(p) directly. For stability, we can use log_sigmoid(x) but we already computed sigmoid.
        # Let's stick to the formula using probabilities for the weight calculation.

        # Weight for positives: (1 - p)^gamma_pos
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)

        pt0 = x_sigmoid_pos
        pt1 = x_sigmoid_neg  # (1 - p)

        # Weights
        pos_weight = pt1.pow(self.gamma_pos)

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)

        # Loss calculation for positives
        # We use -log(p) * weight. To ensure stability, we use -log(p + eps).
        # Alternatively, since x is logits, -log(sigmoid(x)) = -log(1/(1+exp(-x))) = log(1+exp(-x))
        # But we simply use the probabilities calculated:
        loss_pos = -y * pos_weight * torch.log(x_sigmoid_pos.clamp(min=self.eps))

        # --- Negative Part ---
        # L- = (p_m)^gamma_neg * log(1 - p_m)
        # p_m = max(p - clip, 0)

        # Shift probabilities for negatives
        x_sigmoid_neg_shifted = (x_sigmoid_pos - self.clip).clamp(min=0)

        # Weight for negatives: (p_m)^gamma_neg
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)

        neg_weight = x_sigmoid_neg_shifted.pow(self.gamma_neg)

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)

        # Loss calculation for negatives
        # We use -log(1 - p_m) * weight
        # 1 - p_m = 1 - max(p - clip, 0)
        # If p < clip, p_m = 0, 1-p_m = 1, log(1) = 0. Correct.
        loss_neg = (
            -(1 - y)
            * neg_weight
            * torch.log((1 - x_sigmoid_neg_shifted).clamp(min=self.eps))
        )

        # Combine losses
        loss = loss_pos + loss_neg

        # Return mean loss
        return loss.mean()
