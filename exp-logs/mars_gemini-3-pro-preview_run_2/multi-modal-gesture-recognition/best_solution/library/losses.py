import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import get_hyperparams


class ActionSegmentationLoss(nn.Module):
    """
    Implements the composite loss function for the MG-CRGN model with Deep Supervision.

    The loss function aggregates three components across all refinement stages:
    1. Weighted Cross-Entropy Loss (L_cls): Handles class imbalance (0.1 background : 1.0 gestures).
    2. Focal Loss (L_bnd): Focuses on sparse transition frames (boundaries).
    3. Unclamped Probability-Space Smoothing (L_smooth): Enforces temporal consistency using
       MSE on probability differences without clamping, as specified.
    """

    def __init__(self):
        super(ActionSegmentationLoss, self).__init__()
        self.hp = get_hyperparams()

        # Load class weights and register as buffer (automatically moves to device with module)
        # Weights: 0.1 for background (class 0), 1.0 for gestures (classes 1-20)
        weights = torch.tensor(self.hp["class_weights"], dtype=torch.float32)
        if self.hp["device"] == "cuda" and torch.cuda.is_available():
            weights = weights.cuda()
        self.register_buffer("class_weights", weights)

        # Loss coefficients from config
        self.lambda_cls = self.hp["lambda_cls"]
        self.lambda_bnd = self.hp["lambda_bnd"]
        self.lambda_smooth = self.hp["lambda_smooth"]

        # Focal Loss Hyperparameters (Fixed as per standard practice for this task)
        self.focal_gamma = 2.0
        self.focal_alpha = 0.25

    def _calc_focal_loss(self, probs, targets, mask):
        """
        Computes the Masked Focal Loss for boundary detection.

        Args:
            probs (Tensor): [B, T] Predicted boundary probabilities (0.0 to 1.0).
            targets (Tensor): [B, T] Binary boundary targets (0 or 1).
            mask (Tensor): [B, T] Valid frame mask.

        Returns:
            Tensor: Scalar loss value.
        """
        # Clamp probabilities to avoid log(0) numerical instability
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Calculate p_t: probability of the true class
        # If target=1, p_t = p. If target=0, p_t = 1-p.
        p_t = torch.where(targets == 1, probs, 1 - probs)

        # Calculate alpha_t: weighting factor
        # If target=1, alpha_t = alpha. If target=0, alpha_t = 1-alpha.
        alpha_t = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)

        # Focal Loss formula: -alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = -alpha_t * torch.pow(1 - p_t, self.focal_gamma) * torch.log(p_t)

        # Apply mask to ignore padding
        loss = loss * mask

        # Normalize by the number of valid frames
        return loss.sum() / torch.clamp(mask.sum(), min=1.0)

    def _calc_smoothing_loss(self, probs, mask):
        """
        Computes the Unclamped Probability-Space Smoothing Loss.
        This is implemented as the Mean Squared Error (MSE) of temporal differences
        in the probability space, without truncation/clamping.

        Args:
            probs (Tensor): [B, C, T] Class probabilities.
            mask (Tensor): [B, T] Valid frame mask.

        Returns:
            Tensor: Scalar loss value.
        """
        # Calculate temporal difference: P_t - P_{t-1}
        # Slice along time dimension (dim 2)
        diff = probs[:, :, 1:] - probs[:, :, :-1]  # [B, C, T-1]

        # Square the differences (MSE)
        loss_sq = diff.pow(2)

        # Average over classes to get per-frame smoothing cost: [B, T-1]
        loss_mean_cls = loss_sq.mean(dim=1)

        # Create mask for temporal differences (valid if both t and t-1 are valid)
        mask_smooth = mask[:, 1:] * mask[:, :-1]  # [B, T-1]

        # Apply mask
        loss = loss_mean_cls * mask_smooth

        # Normalize
        return loss.sum() / torch.clamp(mask_smooth.sum(), min=1.0)

    def forward(self, stage_outputs, targets, mask):
        """
        Forward pass for the composite loss.

        Args:
            stage_outputs (list[Tensor]): List of outputs from each model stage (Deep Supervision).
                                          Each tensor has shape [B, NumClasses+1, T].
                                          - Channels 0-20: Class Probabilities (Softmax applied).
                                          - Channel 21: Boundary Probability (Sigmoid applied).
            targets (Tensor): [B, T] Ground truth class indices (0-20).
            mask (Tensor): [B, T] Valid frame mask (1.0 for valid, 0.0 for padding).

        Returns:
            Tensor: Total scalar loss.
        """
        total_loss = 0.0

        # Generate Boundary Targets on the fly
        # Boundary = 1 if label changes between t and t+1, else 0.
        # Shift targets to compare t and t+1
        bnd_targets = torch.zeros_like(targets, dtype=torch.float32)
        bnd_targets[:, :-1] = (targets[:, :-1] != targets[:, 1:]).float()

        # Ensure masked regions don't generate false boundaries
        bnd_targets = bnd_targets * mask

        # Iterate over each stage's output for Deep Supervision
        for out in stage_outputs:
            # Split outputs into Classification and Boundary streams
            # Assuming first 21 channels are Class Probs, last channel is Boundary Prob
            cls_probs = out[:, :21, :]  # [B, 21, T]
            bnd_probs = out[:, 21, :]  # [B, T] (Squeeze channel dim)

            # --- 1. Classification Loss (Weighted NLL) ---
            # Inputs are probabilities (Softmaxed), so we take log for NLLLoss.
            cls_log_probs = torch.log(torch.clamp(cls_probs, min=1e-7))

            # Compute NLL Loss per frame
            # reduction='none' returns [B, T] so we can apply the mask manually
            loss_cls_pixel = F.nll_loss(
                cls_log_probs, targets, weight=self.class_weights, reduction="none"
            )

            # Apply mask and normalize
            loss_cls = (loss_cls_pixel * mask).sum() / torch.clamp(mask.sum(), min=1.0)

            # --- 2. Boundary Loss (Focal) ---
            loss_bnd = self._calc_focal_loss(bnd_probs, bnd_targets, mask)

            # --- 3. Smoothing Loss (Unclamped MSE) ---
            loss_smooth = self._calc_smoothing_loss(cls_probs, mask)

            # Aggregate losses for this stage
            total_loss += (
                (self.lambda_cls * loss_cls)
                + (self.lambda_bnd * loss_bnd)
                + (self.lambda_smooth * loss_smooth)
            )

        return total_loss
