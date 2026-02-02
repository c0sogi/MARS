import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import compute_class_weights


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss.
    Enforces temporal smoothness by penalizing rapid changes in probabilities.
    Operates on Softmax probabilities as per instructions to ensure stability.
    """

    def __init__(self, threshold=Config.TMSE_THRESHOLD):
        super(TMSELoss, self).__init__()
        self.threshold = float(threshold)

    def forward(self, probs, mask):
        """
        Args:
            probs: (Batch, Classes, Time) - Softmax probabilities
            mask: (Batch, Time) - Boolean mask indicating valid frames
        Returns:
            loss: Scalar tensor
        """
        # Calculate differences between consecutive frames: P_t - P_{t-1}
        # Shape: (Batch, Classes, Time-1)
        diff = probs[:, :, 1:] - probs[:, :, :-1]

        # Calculate squared differences
        sq_diff = diff.pow(2)

        # Apply truncation: min(diff^2, threshold^2)
        # Note: If operating on probabilities [0,1], max diff^2 is 1.
        # If threshold is > 1 (e.g. 4.0), this acts as standard MSE.
        # We implement the clamp strictly as defined.
        threshold_sq = self.threshold**2
        truncated_diff = torch.clamp(sq_diff, max=threshold_sq)

        # Adjust mask for the diff tensor
        # A transition is valid if both t and t-1 are valid.
        # Since mask is contiguous (1, 1, ..., 0, 0), checking mask[:, 1:] is sufficient.
        # Shape: (Batch, Time-1)
        mask_diff = mask[:, 1:]

        # Expand mask to match classes dimension: (Batch, Classes, Time-1)
        mask_expanded = mask_diff.unsqueeze(1).expand_as(truncated_diff)

        # Apply mask
        masked_loss = truncated_diff * mask_expanded.float()

        # Compute mean over valid elements
        # Sum over all dims, divide by number of valid elements * classes
        valid_elements = mask_expanded.sum()

        if valid_elements > 0:
            loss = masked_loss.sum() / valid_elements
        else:
            loss = torch.tensor(0.0, device=probs.device, requires_grad=True)

        return loss


class MultiStageLoss(nn.Module):
    """
    Aggregated loss function for the IDC-RCN model.
    Computes Weighted Cross-Entropy for all stages and T-MSE for refinement stages.
    """

    def __init__(self):
        super(MultiStageLoss, self).__init__()

        # Load class weights
        self.weights = compute_class_weights()

        # Initialize Cross Entropy Loss
        # We do not use reduction='mean' directly because we need to handle masking manually
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction="none")

        # Initialize Smoothing Loss
        self.tmse_loss = TMSELoss(threshold=Config.TMSE_THRESHOLD)

        self.lambda_tmse = Config.LAMBDA_TMSE

    def forward(self, model_outputs, targets, mask):
        """
        Args:
            model_outputs: dict containing 'stage1', 'stage2', 'stage3' outputs
                           Each is (Batch, Classes, Time) - Logits or Probs?
                           Note: The model outputs from model.py are PROBABILITIES (Softmax applied).
                           However, nn.CrossEntropyLoss expects Logits.
                           We need to be careful.
                           Looking at model.py:
                           Stage 1 (BiLSTM): returns `probs` (Softmax applied).
                           Stage 2/3 (TCN): returns `probs` (Softmax applied).

                           Since the model returns probabilities, we must use NLLLoss on log(probs)
                           or convert back to logits (unstable).
                           Ideally, we take log(probs) and use NLLLoss.
            targets: (Batch, Time) - Ground truth labels
            mask: (Batch, Time) - Boolean mask

        Returns:
            total_loss: Scalar
            metrics: dict of individual loss components
        """

        # Extract outputs
        p0 = model_outputs["stage1"]
        p1 = model_outputs["stage2"]
        p2 = model_outputs["stage3"]

        # --- Cross Entropy Loss Calculation ---
        # Since inputs are probabilities, we use NLLLoss on log(probs) + epsilon
        # This is mathematically equivalent to CrossEntropy on logits

        eps = 1e-8

        def compute_masked_ce(probs, targets, mask):
            # probs: (B, C, T)
            # targets: (B, T)
            # mask: (B, T)

            # Permute to (B, T, C)
            probs_perm = probs.permute(0, 2, 1)

            # Flatten
            probs_flat = probs_perm.reshape(-1, Config.NUM_CLASSES)
            targets_flat = targets.reshape(-1)
            mask_flat = mask.reshape(-1)

            # Filter valid elements
            valid_probs = probs_flat[mask_flat]
            valid_targets = targets_flat[mask_flat]

            if valid_probs.size(0) == 0:
                return torch.tensor(0.0, device=probs.device, requires_grad=True)

            # Compute Log Probs
            log_probs = torch.log(valid_probs + eps)

            # Compute NLL Loss with weights
            # F.nll_loss expects log_probs
            loss = F.nll_loss(
                log_probs, valid_targets, weight=self.weights, reduction="mean"
            )

            return loss

        loss_ce_0 = compute_masked_ce(p0, targets, mask)
        loss_ce_1 = compute_masked_ce(p1, targets, mask)
        loss_ce_2 = compute_masked_ce(p2, targets, mask)

        # --- T-MSE Smoothing Loss Calculation ---
        # Applied to Stage 2 and Stage 3 probabilities
        loss_tmse_1 = self.tmse_loss(p1, mask)
        loss_tmse_2 = self.tmse_loss(p2, mask)

        # --- Total Loss Aggregation ---
        # Stage 1: Only CE
        l1 = loss_ce_0

        # Stage 2: CE + Lambda * TMSE
        l2 = loss_ce_1 + self.lambda_tmse * loss_tmse_1

        # Stage 3: CE + Lambda * TMSE
        l3 = loss_ce_2 + self.lambda_tmse * loss_tmse_2

        total_loss = l1 + l2 + l3

        metrics = {
            "loss_stage1": l1.item(),
            "loss_stage2": l2.item(),
            "loss_stage3": l3.item(),
            "ce_0": loss_ce_0.item(),
            "ce_1": loss_ce_1.item(),
            "ce_2": loss_ce_2.item(),
            "tmse_1": loss_tmse_1.item(),
            "tmse_2": loss_tmse_2.item(),
        }

        return total_loss, metrics
