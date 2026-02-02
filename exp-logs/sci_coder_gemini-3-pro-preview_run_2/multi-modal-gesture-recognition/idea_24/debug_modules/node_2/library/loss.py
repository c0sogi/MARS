import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    CLASS_WEIGHTS,
    LOSS_WEIGHT_CLS,
    LOSS_WEIGHT_BND,
    LOSS_WEIGHT_SMOOTH,
)


class ActionSegmentationLoss(nn.Module):
    """
    Implements the Multi-Stage Deep Supervision loss for the SBG-CRCN model.
    Aggregates:
    1. Weighted Cross-Entropy for classification.
    2. MSE for soft-boundary regression.
    3. Unclamped MSE (T-MSE) for probability smoothing.
    """

    def __init__(self):
        super(ActionSegmentationLoss, self).__init__()
        # Register class weights as a buffer to ensure they move to the correct device
        self.register_buffer("class_weights", CLASS_WEIGHTS)

        # Loss component weights
        self.lambda_cls = LOSS_WEIGHT_CLS
        self.lambda_bnd = LOSS_WEIGHT_BND
        self.lambda_smooth = LOSS_WEIGHT_SMOOTH

        # Epsilon for numerical stability in log
        self.eps = 1e-7

    def compute_cls_loss(self, probs, targets, mask):
        """
        Computes Weighted Cross-Entropy (NLL) Loss.
        Args:
            probs: (B, T, C) Softmax probabilities.
            targets: (B, T) Integer class labels.
            mask: (B, T) Boolean mask.
        """
        # Flatten dimensions
        B, T, C = probs.shape
        probs_flat = probs.view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Select valid frames
        valid_probs = probs_flat[mask_flat]
        valid_targets = targets_flat[mask_flat]

        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=probs.device)

        # Compute Log-Probabilities manually since input is Softmax output
        log_probs = torch.log(valid_probs + self.eps)

        # NLLLoss with class weights
        loss = F.nll_loss(
            log_probs, valid_targets, weight=self.class_weights, reduction="mean"
        )
        return loss

    def compute_bnd_loss(self, probs, targets, mask):
        """
        Computes Mean Squared Error for Boundary Regression.
        Args:
            probs: (B, T, 1) Sigmoid probabilities.
            targets: (B, T) Gaussian soft targets.
            mask: (B, T) Boolean mask.
        """
        probs_flat = probs.squeeze(-1).view(-1)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        valid_probs = probs_flat[mask_flat]
        valid_targets = targets_flat[mask_flat]

        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=probs.device)

        loss = F.mse_loss(valid_probs, valid_targets, reduction="mean")
        return loss

    def compute_smooth_loss(self, probs, mask):
        """
        Computes Probability-Space Smoothing Loss (Unclamped T-MSE).
        Minimizes squared difference between adjacent frame probabilities.
        Args:
            probs: (B, T, C) Softmax probabilities.
            mask: (B, T) Boolean mask.
        """
        # Calculate difference between t and t-1
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Create mask for transitions (both t and t-1 must be valid)
        # Shape: (B, T-1)
        valid_mask = mask[:, 1:] & mask[:, :-1]

        # Select valid differences
        diff_flat = diff[valid_mask]

        if diff_flat.numel() == 0:
            return torch.tensor(0.0, device=probs.device)

        # Mean Squared Error on probability differences
        loss = torch.mean(diff_flat**2)
        return loss

    def forward(self, model_outputs, class_targets, boundary_targets, mask):
        """
        Calculates the total loss across all stages.

        Args:
            model_outputs (dict): Dictionary of stage outputs. Each value is a dict
                                  with 'class_probs' and 'boundary_probs'.
            class_targets (torch.Tensor): (B, T) Ground truth class labels.
            boundary_targets (torch.Tensor): (B, T) Ground truth soft boundaries.
            mask (torch.Tensor): (B, T) Sequence mask.

        Returns:
            total_loss (torch.Tensor): Scalar loss for backpropagation.
            metrics (dict): Dictionary of detached loss components for logging.
        """
        total_loss = 0.0
        metrics = {}

        # Iterate through all stages (stage1, stage2, stage3)
        for stage_name, stage_out in model_outputs.items():
            cls_probs = stage_out["class_probs"]
            bnd_probs = stage_out["boundary_probs"]

            # 1. Classification Loss
            l_cls = self.compute_cls_loss(cls_probs, class_targets, mask)

            # 2. Boundary Loss
            l_bnd = self.compute_bnd_loss(bnd_probs, boundary_targets, mask)

            # 3. Smoothness Loss
            l_smooth = self.compute_smooth_loss(cls_probs, mask)

            # Weighted Sum for this stage
            stage_loss = (
                (self.lambda_cls * l_cls)
                + (self.lambda_bnd * l_bnd)
                + (self.lambda_smooth * l_smooth)
            )

            total_loss += stage_loss

            # Log metrics
            metrics[f"{stage_name}_loss"] = stage_loss.item()
            metrics[f"{stage_name}_cls"] = l_cls.item()
            metrics[f"{stage_name}_bnd"] = l_bnd.item()
            metrics[f"{stage_name}_smooth"] = l_smooth.item()

        return total_loss, metrics
