import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (TMSE) Loss for temporal smoothing.

    As per the task instructions:
    - Applied to Softmax probabilities.
    - "Unclamped": We do not apply a truncation threshold (or effectively threshold=infinity).
    - "Not conditioned on boundary": We compute smoothness over the raw probability sequence.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()

    def forward(self, logits, mask=None):
        """
        Args:
            logits: (Batch, Time, Classes)
            mask: (Batch, Time) - Boolean mask indicating valid frames.
        Returns:
            Scalar loss.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute differences between adjacent frames: P_t - P_{t-1}
        # Shape: (Batch, Time-1, Classes)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared Error
        mse = diff.pow(2)

        # Apply Masking
        if mask is not None:
            # Adjust mask for the reduced time dimension (T-1)
            # We use mask[:, 1:] because if frame t is padding, diff(t, t-1) is invalid.
            # Actually, if frame t is valid and t-1 is valid, the transition is valid.
            # If mask is 1 for valid, 0 for pad.
            # mask_t: (B, T-1)
            mask_t = mask[:, 1:].unsqueeze(-1)  # (Batch, Time-1, 1)

            # Apply mask
            mse = mse * mask_t

            # Normalize by number of valid transitions
            # Sum over all dims, divide by sum of mask * classes
            valid_elements = mask_t.sum() * mse.size(-1)
            if valid_elements > 0:
                loss = mse.sum() / valid_elements
            else:
                loss = torch.tensor(0.0, device=probs.device)
        else:
            loss = mse.mean()

        return loss


class DeepSupervisionLoss(nn.Module):
    """
    Composite loss function for the DSL-CRCN architecture.
    Aggregates Weighted Cross-Entropy and TMSE Loss across three stages.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

        # Class Weights for CrossEntropy
        self.class_weights = Config.get_class_weights_tensor()

        # Base Losses
        # reduction='none' to allow manual masking
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, reduction="none"
        )
        self.tmse_criterion = TMSELoss()

        self.tmse_weight = Config.TMSE_WEIGHT
        self.num_classes = Config.NUM_CLASSES

    def forward(self, model_outputs, targets, lengths):
        """
        Args:
            model_outputs: Tuple (stage1_out, stage2_out, stage3_out)
                - Stage 1 & 2: (Batch, Time, Classes + 1) [Class Probs + Transition]
                - Stage 3: (Batch, Time, Classes)
            targets: (Batch, Time) - Ground truth labels
            lengths: (Batch,) - Sequence lengths

        Returns:
            total_loss: Scalar
            metrics: Dictionary of individual loss components for logging
        """
        stage1_out, stage2_out, stage3_out = model_outputs

        # Generate Mask
        # shape: (Batch, Time)
        max_len = targets.size(1)
        batch_size = targets.size(0)
        device = targets.device

        # Create mask: True for valid frames, False for padding
        idx_range = (
            torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        mask = idx_range < lengths.unsqueeze(1)

        # Helper to compute stage loss
        def compute_stage_loss(logits_full, is_intermediate=False):
            # If intermediate stage, slice off the transition channel
            if is_intermediate:
                # Assuming the first NUM_CLASSES channels are class predictions
                # and the last channel is the transition head.
                logits_cls = logits_full[:, :, : self.num_classes]
            else:
                logits_cls = logits_full

            # 1. Cross Entropy Loss
            # Flatten for CE: (B*T, C) and (B*T)
            # But we need masking.
            # CE with reduction='none' returns (B, T)
            # We need to permute logits to (B, C, T) for CE Loss expects (B, C, ...)
            ce_loss_raw = self.ce_criterion(logits_cls.transpose(1, 2), targets)

            # Apply mask
            ce_loss = (ce_loss_raw * mask).sum() / (mask.sum() + 1e-8)

            # 2. TMSE Loss
            tmse_loss = self.tmse_criterion(logits_cls, mask)

            return ce_loss, tmse_loss

        # Compute losses for each stage
        ce1, tmse1 = compute_stage_loss(stage1_out, is_intermediate=True)
        ce2, tmse2 = compute_stage_loss(stage2_out, is_intermediate=True)
        ce3, tmse3 = compute_stage_loss(stage3_out, is_intermediate=False)

        # Weighted Sum
        # L_total = L_stage1 + L_stage2 + L_stage3
        # L_stage = CE + lambda * TMSE

        loss_stage1 = ce1 + self.tmse_weight * tmse1
        loss_stage2 = ce2 + self.tmse_weight * tmse2
        loss_stage3 = ce3 + self.tmse_weight * tmse3

        total_loss = loss_stage1 + loss_stage2 + loss_stage3

        metrics = {
            "loss": total_loss.item(),
            "ce1": ce1.item(),
            "ce2": ce2.item(),
            "ce3": ce3.item(),
            "tmse1": tmse1.item(),
            "tmse2": tmse2.item(),
            "tmse3": tmse3.item(),
        }

        return total_loss, metrics
