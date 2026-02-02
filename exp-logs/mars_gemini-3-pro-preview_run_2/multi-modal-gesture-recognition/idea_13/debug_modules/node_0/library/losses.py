import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CombinedLoss(nn.Module):
    """
    Implements the multi-objective loss for the Supervised Boundary-Aware Masked
    Dual-Stage Cascaded Recurrent-Convolutional Network (SB-MD-CRCN).

    Objectives:
    1. Weighted Cross-Entropy for Classification (Stages 1, 2, 3)
    2. Weighted Binary Cross-Entropy for Boundary Detection (Stages 1, 2, 3)
    3. Temporal Smoothing (MSE on Probabilities) (Stages 2, 3)
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        # Load configuration
        self.num_classes = Config.NUM_CLASSES

        # Class Weights: 0.1 for background, 1.0 for others
        # We register as buffer to ensure it moves to device with the module
        self.register_buffer("class_weights", Config.CLASS_WEIGHTS)

        # Boundary Positive Weight
        self.register_buffer(
            "boundary_pos_weight", torch.tensor(Config.BOUNDARY_POS_WEIGHT)
        )

        # Loss Coefficients
        self.lambda_cls = Config.LOSS_LAMBDA_CLS
        self.lambda_bnd = Config.LOSS_LAMBDA_BND
        self.lambda_smooth = Config.LOSS_LAMBDA_SMOOTH

    def calc_masked_weighted_ce(self, logits, targets, mask):
        """
        Computes masked weighted Cross Entropy Loss.

        Args:
            logits: (B, T, C)
            targets: (B, T)
            mask: (B, T) - Boolean or Float mask (1 for valid)
        """
        # Flatten for CrossEntropyLoss
        B, T, C = logits.shape
        logits_flat = logits.view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Compute element-wise loss
        loss = F.cross_entropy(
            logits_flat, targets_flat, weight=self.class_weights, reduction="none"
        )

        # Apply mask
        masked_loss = loss * mask_flat

        # Normalize by number of valid tokens
        num_valid = mask_flat.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum() * 0.0

    def calc_masked_weighted_bce(self, logits, targets, mask):
        """
        Computes masked weighted Binary Cross Entropy Loss for boundaries.

        Args:
            logits: (B, T, 1) or (B, T)
            targets: (B, T) - 0 or 1
            mask: (B, T)
        """
        # Ensure logits and targets match shapes
        if logits.dim() == 3 and logits.size(2) == 1:
            logits = logits.squeeze(2)

        targets = targets.float()

        # Flatten
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Compute element-wise loss with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits_flat,
            targets_flat,
            pos_weight=self.boundary_pos_weight,
            reduction="none",
        )

        # Apply mask
        masked_loss = loss * mask_flat

        # Normalize
        num_valid = mask_flat.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum() * 0.0

    def calc_masked_tmse(self, logits, mask):
        """
        Computes Temporal Smoothing Loss (T-MSE) on Softmax probabilities.
        L = Mean( || Softmax(t) - Softmax(t-1) ||^2 )

        Args:
            logits: (B, T, C)
            mask: (B, T)
        """
        probs = F.softmax(logits, dim=-1)  # (B, T, C)

        # Calculate difference between t and t-1
        # diff[t] = probs[t+1] - probs[t]
        # We slice from 1: to end and 0: to end-1
        curr_probs = probs[:, 1:, :]
        prev_probs = probs[:, :-1, :]

        diff = curr_probs - prev_probs
        squared_diff = torch.sum(diff**2, dim=-1)  # (B, T-1)

        # Adjust mask for the shortened sequence
        # A transition (t-1 -> t) is valid only if both t-1 and t are valid.
        # Assuming contiguous mask (1s then 0s), mask[:, 1:] is sufficient.
        mask_sliced = mask[:, 1:]

        masked_loss = squared_diff * mask_sliced

        num_valid = mask_sliced.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum() * 0.0

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                'stage1_cls', 'stage1_bnd',
                'stage2_cls', 'stage2_bnd',
                'stage3_cls', 'stage3_bnd'
            targets (dict): Dictionary containing targets:
                'cls_targets': (B, T) LongTensor
                'bnd_targets': (B, T) FloatTensor
                'mask': (B, T) Bool/FloatTensor

        Returns:
            dict: Dictionary containing 'loss' (total) and individual components for logging.
        """
        cls_targets = targets["cls_targets"]
        bnd_targets = targets["bnd_targets"]
        mask = targets["mask"]

        total_loss = 0.0
        loss_dict = {}

        # --- Stage 1 ---
        # Classification
        s1_cls_loss = self.calc_masked_weighted_ce(
            outputs["stage1_cls"], cls_targets, mask
        )
        # Boundary
        s1_bnd_loss = self.calc_masked_weighted_bce(
            outputs["stage1_bnd"], bnd_targets, mask
        )

        stage1_loss = (self.lambda_cls * s1_cls_loss) + (self.lambda_bnd * s1_bnd_loss)
        total_loss += stage1_loss

        loss_dict["s1_cls"] = s1_cls_loss.item()
        loss_dict["s1_bnd"] = s1_bnd_loss.item()

        # --- Stage 2 ---
        # Classification
        s2_cls_loss = self.calc_masked_weighted_ce(
            outputs["stage2_cls"], cls_targets, mask
        )
        # Boundary
        s2_bnd_loss = self.calc_masked_weighted_bce(
            outputs["stage2_bnd"], bnd_targets, mask
        )
        # Smoothing (T-MSE)
        s2_smooth_loss = self.calc_masked_tmse(outputs["stage2_cls"], mask)

        stage2_loss = (
            (self.lambda_cls * s2_cls_loss)
            + (self.lambda_bnd * s2_bnd_loss)
            + (self.lambda_smooth * s2_smooth_loss)
        )
        total_loss += stage2_loss

        loss_dict["s2_cls"] = s2_cls_loss.item()
        loss_dict["s2_bnd"] = s2_bnd_loss.item()
        loss_dict["s2_smooth"] = s2_smooth_loss.item()

        # --- Stage 3 ---
        # Classification
        s3_cls_loss = self.calc_masked_weighted_ce(
            outputs["stage3_cls"], cls_targets, mask
        )
        # Boundary
        s3_bnd_loss = self.calc_masked_weighted_bce(
            outputs["stage3_bnd"], bnd_targets, mask
        )
        # Smoothing (T-MSE)
        s3_smooth_loss = self.calc_masked_tmse(outputs["stage3_cls"], mask)

        stage3_loss = (
            (self.lambda_cls * s3_cls_loss)
            + (self.lambda_bnd * s3_bnd_loss)
            + (self.lambda_smooth * s3_smooth_loss)
        )
        total_loss += stage3_loss

        loss_dict["s3_cls"] = s3_cls_loss.item()
        loss_dict["s3_bnd"] = s3_bnd_loss.item()
        loss_dict["s3_smooth"] = s3_smooth_loss.item()

        loss_dict["loss"] = total_loss

        return loss_dict
