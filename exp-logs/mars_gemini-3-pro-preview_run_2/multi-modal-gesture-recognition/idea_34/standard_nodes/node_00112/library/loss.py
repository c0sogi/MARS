import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import HYPERPARAMS


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for probability-space smoothing.
    Computes MSE between adjacent frame probabilities to encourage temporal smoothness.

    As per instructions:
    - Operates on Softmax probabilities (not Log-Softmax).
    - "Unclamped": We use a threshold of 0.0, effectively making it MSE.
    """

    def __init__(self, threshold=0.0):
        super(TMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, probs, mask):
        """
        Args:
            probs: (B, T, C) - Softmax probabilities
            mask: (B, T) - Sequence mask (1 for valid, 0 for pad)
        Returns:
            loss: scalar
        """
        # Calculate differences between adjacent frames: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared difference
        squared_diff = diff**2

        # Apply truncation (if threshold > 0)
        if self.threshold > 0:
            squared_diff = torch.clamp(squared_diff - self.threshold, min=0.0)

        # Masking
        # We need a mask for the transitions.
        # A transition t -> t+1 is valid if both t and t+1 are valid.
        # mask is (B, T). mask[:, 1:] corresponds to t+1, mask[:, :-1] to t.
        mask_transitions = mask[:, 1:] * mask[:, :-1]  # (B, T-1)

        # Expand mask for channels: (B, T-1, 1)
        mask_transitions = mask_transitions.unsqueeze(-1)

        # Apply mask
        masked_loss = squared_diff * mask_transitions

        # Normalize by total valid transitions * channels
        total_transitions = mask_transitions.sum() * probs.shape[2]

        if total_transitions == 0:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        return masked_loss.sum() / total_transitions


class DeepSupervisionLoss(nn.Module):
    """
    Composite loss function for RLSG-CN with Deep Supervision.
    Aggregates Classification, Boundary, and Smoothing losses across all stages.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

        self.hp = HYPERPARAMS
        weights = self.hp["loss_weights"]
        self.w_cls = weights["cls"]
        self.w_bnd = weights["bnd"]
        self.w_smooth = weights["smooth"]

        # Class Weights for CrossEntropy
        # Convert list to tensor
        class_weights_tensor = torch.tensor(
            self.hp["class_weights"], dtype=torch.float32
        )
        # We register it as a buffer so it moves to device automatically with the module
        self.register_buffer("class_weights", class_weights_tensor)

        # Loss Components
        # Reduction='none' to allow manual masking
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.tmse_loss = TMSELoss(threshold=0.0)

    def forward(self, model_outputs, targets, mask):
        """
        Args:
            model_outputs: Dict containing outputs for 'stage1', 'stage2', 'stage3'
            targets: Dict containing 'cls_target' (B, T) and 'bnd_target' (B, T)
            mask: (B, T) - Sequence mask
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss components for logging
        """
        total_loss = 0.0
        loss_dict = {}

        cls_target = targets["cls_target"]  # (B, T)
        bnd_target = targets["bnd_target"]  # (B, T)

        # Calculate number of valid elements for normalization
        num_valid = mask.sum()
        if num_valid == 0:
            # Should not happen with proper batching
            num_valid = 1.0

        # Iterate over all stages present in outputs
        for stage_name, stage_out in model_outputs.items():
            # 1. Classification Loss (Cross Entropy)
            # logits: (B, T, C) -> Need (B, C, T) for CrossEntropyLoss
            cls_logits = stage_out["cls_logits"].transpose(1, 2)
            ce = self.ce_loss(cls_logits, cls_target)  # (B, T)
            masked_ce = (ce * mask).sum() / num_valid

            # 2. Boundary Loss (Binary Cross Entropy)
            # logits: (B, T, 1) -> Squeeze to (B, T)
            bnd_logits = stage_out["bnd_logits"].squeeze(-1)
            bce = self.bce_loss(bnd_logits, bnd_target)  # (B, T)
            masked_bce = (bce * mask).sum() / num_valid

            # 3. Smoothing Loss (T-MSE)
            # Operates on probabilities: (B, T, C)
            cls_probs = stage_out["cls_probs"]
            tmse = self.tmse_loss(cls_probs, mask)

            # Weighted Sum for this stage
            stage_loss = (
                (self.w_cls * masked_ce)
                + (self.w_bnd * masked_bce)
                + (self.w_smooth * tmse)
            )

            total_loss += stage_loss

            # Log components
            loss_dict[f"{stage_name}_loss"] = stage_loss.item()
            loss_dict[f"{stage_name}_ce"] = masked_ce.item()
            loss_dict[f"{stage_name}_bnd"] = masked_bce.item()
            loss_dict[f"{stage_name}_smooth"] = tmse.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
