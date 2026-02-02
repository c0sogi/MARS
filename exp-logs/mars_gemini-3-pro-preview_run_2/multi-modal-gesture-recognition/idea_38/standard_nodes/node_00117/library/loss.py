import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for boundary detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets, mask):
        """
        Args:
            probs (torch.Tensor): (B, T, 1) Sigmoid probabilities.
            targets (torch.Tensor): (B, T) Binary targets (0 or 1).
            mask (torch.Tensor): (B, T) Boolean mask.
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Flatten and mask
        # probs: (B, T, 1) -> (N,)
        valid_probs = probs.squeeze(-1)[mask]
        valid_targets = targets[mask].float()

        # Numerical stability
        eps = 1e-7
        valid_probs = torch.clamp(valid_probs, eps, 1.0 - eps)

        # Calculate p_t (probability of the true class)
        # If y=1, p_t = p. If y=0, p_t = 1-p.
        p_t = torch.where(valid_targets == 1, valid_probs, 1 - valid_probs)

        # Calculate alpha_t
        # If y=1, alpha_t = alpha. If y=0, alpha_t = 1-alpha.
        alpha_t = torch.where(valid_targets == 1, self.alpha, 1 - self.alpha)

        # Focal Loss
        loss = -alpha_t * torch.pow(1 - p_t, self.gamma) * torch.log(p_t)

        return loss.mean()


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error Loss for temporal smoothing.
    L = mean( min( (P_t - P_{t-1})^2, threshold^2 ) )
    """

    def __init__(self, threshold=4.0):
        super(TMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, probs, mask):
        """
        Args:
            probs (torch.Tensor): (B, T, C) Softmax probabilities.
            mask (torch.Tensor): (B, T) Boolean mask.
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Check sequence length
        if probs.shape[1] <= 1:
            return torch.tensor(0.0, device=probs.device)

        # Calculate diffs: P_t - P_{t-1}
        # probs[:, 1:, :] corresponds to t
        # probs[:, :-1, :] corresponds to t-1
        diff = probs[:, 1:, :] - probs[:, :-1, :]
        sq_diff = torch.pow(diff, 2)

        # Sum over classes (C)
        mse_per_step = torch.sum(sq_diff, dim=2)  # (B, T-1)

        # Apply Truncation (Clamp)
        # Note: With threshold=4.0 and probs in [0,1], this is effectively unclamped
        truncated_mse = torch.clamp(mse_per_step, max=self.threshold**2)

        # Apply Mask
        # A step is valid if both t and t-1 are valid
        valid_mask = mask[:, 1:] & mask[:, :-1]

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)

        loss = truncated_mse[valid_mask].mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Multi-Objective Loss Function for HCRG-CN.
    Aggregates Classification, Boundary, and Smoothing losses across all stages.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        # Load Weights from Config
        self.class_weights = Config.CLASS_WEIGHTS_TENSOR
        self.w_cls = Config.W_CLS
        self.w_bnd = Config.W_BND
        self.w_smooth = Config.W_SMOOTH

        # Initialize Components
        self.focal_loss = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.tmse_loss = TMSELoss(threshold=Config.TMSE_THRESHOLD)

        # Classification Loss (NLLLoss expects log-probabilities)
        self.nll_loss = nn.NLLLoss(weight=self.class_weights, reduction="mean")

    def forward(self, predictions, targets, boundaries, mask):
        """
        Args:
            predictions (dict): Output from HCRGCN model.
                                {'stage1': (cls, bnd), 'stage2': ...}
            targets (torch.Tensor): (B, T) Class indices.
            boundaries (torch.Tensor): (B, T) Boundary labels (0.0 or 1.0).
            mask (torch.Tensor): (B, T) Boolean mask.

        Returns:
            torch.Tensor: Total combined loss.
        """
        total_loss = 0.0

        # Ensure class weights are on the correct device
        if self.nll_loss.weight.device != targets.device:
            self.nll_loss.weight = self.nll_loss.weight.to(targets.device)

        # Iterate through stages
        for stage_name in ["stage1", "stage2", "stage3"]:
            if stage_name not in predictions:
                continue

            cls_probs, bnd_probs = predictions[stage_name]

            # 1. Classification Loss (Weighted Cross Entropy)
            # Flatten and select valid elements
            active_cls_probs = cls_probs[mask]  # (N, C)
            active_targets = targets[mask]  # (N,)

            # Convert Softmax Probs to Log Probs for NLLLoss
            eps = 1e-7
            log_probs = torch.log(active_cls_probs + eps)

            loss_cls = self.nll_loss(log_probs, active_targets)

            # 2. Boundary Loss (Focal Loss)
            loss_bnd = self.focal_loss(bnd_probs, boundaries, mask)

            # 3. Smoothing Loss (T-MSE)
            loss_smooth = self.tmse_loss(cls_probs, mask)

            # Aggregate Stage Loss
            stage_loss = (
                (self.w_cls * loss_cls)
                + (self.w_bnd * loss_bnd)
                + (self.w_smooth * loss_smooth)
            )

            total_loss += stage_loss

        return total_loss
