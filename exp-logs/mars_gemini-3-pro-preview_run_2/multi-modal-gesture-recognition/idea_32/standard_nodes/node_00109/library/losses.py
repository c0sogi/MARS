import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as config


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Segmentation Loss for BMGCN.

    Aggregates losses from three components across multiple stages:
    1. Weighted Cross Entropy (Classification)
    2. Boundary Binary Cross Entropy (Boundary Detection)
    3. Truncated Mean Squared Error (Smoothness)
    """

    def __init__(self, train_config=config.TRAIN_CONFIG):
        super(CombinedSegmentationLoss, self).__init__()
        self.config = train_config

        # --- 1. Classification Loss Setup ---
        # Weights: 0.1 for background (index 0), 1.0 for gestures (indices 1-20)
        weights = self.config.get("class_weights", [0.1] + [1.0] * 20)
        # We convert to tensor here; nn.CrossEntropyLoss will handle device movement
        self.class_weights = torch.tensor(weights).float()

        self.criterion_cls = nn.CrossEntropyLoss(
            weight=self.class_weights, reduction="none"
        )

        # --- 2. Boundary Loss Setup ---
        self.criterion_bnd = nn.BCEWithLogitsLoss(reduction="none")

        # --- 3. Smoothness Loss Setup ---
        # Truncated MSE Threshold.
        # If the squared difference between frames exceeds this, it is clamped.
        # This allows for sharp transitions (boundaries) without incurring massive loss.
        # 0.04 corresponds to a probability jump of 0.2.
        self.tmse_threshold = 0.04

        # Loss Component Weights
        self.lambda_cls = self.config.get("lambda_cls", 1.0)
        self.lambda_bnd = self.config.get("lambda_bnd", 1.0)
        self.lambda_smooth = self.config.get("lambda_smooth", 0.15)

    def forward(self, outputs, targets):
        """
        Computes the weighted sum of losses across all stages.

        Args:
            outputs (dict): Dictionary of model outputs per stage (e.g., 'stage1', 'stage2').
                            Each stage dict contains 'cls_logits', 'cls_probs', 'bnd_logits'.
            targets (dict): Dictionary containing 'cls_labels', 'bnd_labels', 'mask'.

        Returns:
            total_loss (Tensor): Scalar loss for backpropagation.
            stats (dict): Dictionary of loss components for logging.
        """
        cls_target = targets["cls_labels"]
        bnd_target = targets["bnd_labels"].unsqueeze(-1)  # (B, T, 1)
        mask = targets["mask"]

        total_loss = 0.0
        stats = {}

        # Iterate over all available stages in the output (Stage 1, 2, 3)
        for stage_name, out in outputs.items():

            # --- 1. Classification Loss (Weighted CE) ---
            # Reshape (B, T, C) -> (B*T, C) for CrossEntropyLoss
            B, T, C = out["cls_logits"].shape
            cls_logits_flat = out["cls_logits"].reshape(-1, C)
            cls_target_flat = cls_target.reshape(-1)

            loss_cls_raw = self.criterion_cls(cls_logits_flat, cls_target_flat)
            loss_cls_raw = loss_cls_raw.view(B, T)

            # Apply Mask
            valid_elements = mask.sum() + 1e-6
            loss_cls = (loss_cls_raw * mask).sum() / valid_elements

            # --- 2. Boundary Loss (BCE) ---
            loss_bnd_raw = self.criterion_bnd(out["bnd_logits"], bnd_target)
            loss_bnd = (loss_bnd_raw.squeeze(-1) * mask).sum() / valid_elements

            # --- 3. Smoothness Loss (T-MSE) ---
            # Use Softmax probabilities for smoothness
            probs = out["cls_probs"]  # (B, T, C)

            # Calculate temporal difference: P_t - P_{t-1}
            # Slice to get t=1..T and t=0..T-1
            diff = probs[:, 1:, :] - probs[:, :-1, :]

            # Mean Squared Error per frame (average over classes)
            mse = torch.mean(diff**2, dim=-1)  # (B, T-1)

            # Truncate (Clamp) the error to allow sharp transitions
            tmse = torch.clamp(mse, max=self.tmse_threshold)

            # Mask for smoothness (requires both t and t-1 to be valid)
            mask_smooth = mask[:, 1:] * mask[:, :-1]
            loss_smooth = (tmse * mask_smooth).sum() / (mask_smooth.sum() + 1e-6)

            # --- Aggregation ---
            stage_loss = (
                self.lambda_cls * loss_cls
                + self.lambda_bnd * loss_bnd
                + self.lambda_smooth * loss_smooth
            )

            total_loss += stage_loss

            # Logging Stats
            stats[f"{stage_name}_loss"] = stage_loss.item()
            stats[f"{stage_name}_cls"] = loss_cls.item()
            stats[f"{stage_name}_bnd"] = loss_bnd.item()
            stats[f"{stage_name}_smooth"] = loss_smooth.item()

        return total_loss, stats
