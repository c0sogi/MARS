import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config


def get_pos_weights(load_cached_data=True):
    """
    Calculates or loads positive weights for BCEWithLogitsLoss based on class imbalance.
    Formula: weight_pos = number_of_negatives / number_of_positives

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        np.ndarray: Array of weights for each class.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "pos_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return weights
        except Exception:
            pass

    # 2. Compute from scratch
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(Config.TRAIN_METADATA):
        df = pd.read_csv(Config.TRAIN_METADATA)
        targets = df[Config.TARGET_COLS].values

        # Calculate weights per class
        # Add epsilon to avoid division by zero
        pos_counts = np.sum(targets, axis=0)
        total_counts = len(targets)
        neg_counts = total_counts - pos_counts

        weights = neg_counts / (pos_counts + 1e-6)

        # Save to cache
        np.save(cache_path, weights)
        return weights
    else:
        # Fallback if metadata is missing (e.g. inference only env)
        # Return ones
        return np.ones(len(Config.TARGET_COLS))


class MultiTaskLoss(nn.Module):
    """
    Composite loss function for Multi-Task Learning.
    Combines Classification Loss (Weighted BCE) and Auxiliary Segmentation Loss (BCE).
    """

    def __init__(self, load_cached_data=True):
        super().__init__()

        # Load class weights for classification imbalance
        weights_np = get_pos_weights(load_cached_data=load_cached_data)
        self.pos_weights = torch.tensor(weights_np, dtype=torch.float32).to(
            Config.DEVICE
        )

        # Classification Loss: BCE with class weights
        self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        # Segmentation Loss: BCE (pixel-wise)
        # We use reduction='none' to manually handle the mask validity flag
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.seg_weight = Config.SEG_LOSS_WEIGHT

    def forward(self, cls_logits, cls_targets, seg_logits, seg_targets, mask_validity):
        """
        Args:
            cls_logits: (B, NumClasses)
            cls_targets: (B, NumClasses)
            seg_logits: (B, 1, H, W)
            seg_targets: (B, 1, H, W)
            mask_validity: (B,) - 1.0 if seg target is valid, 0.0 otherwise

        Returns:
            dict: Dictionary containing total loss and individual components.
        """
        # 1. Classification Loss
        cls_loss = self.cls_criterion(cls_logits, cls_targets)

        # 2. Segmentation Loss (Auxiliary)
        # Calculate pixel-wise loss
        pixel_loss = self.seg_criterion(seg_logits, seg_targets)

        # Average over spatial dimensions (H, W) and channel to get per-sample loss
        # pixel_loss: (B, 1, H, W) -> sample_loss: (B,)
        sample_loss = pixel_loss.mean(dim=(1, 2, 3))

        # Apply validity mask
        # Only compute gradients for samples that have ground truth segmentation
        if mask_validity.ndim > 1:
            mask_validity = mask_validity.view(-1)

        valid_loss = sample_loss * mask_validity

        # Average over the batch, considering only valid samples
        num_valid = mask_validity.sum()

        if num_valid > 0:
            seg_loss = valid_loss.sum() / num_valid
        else:
            # If no valid masks in batch, seg_loss is 0 but must maintain graph connectivity if needed
            # (though usually detached is fine, we use 0 tensor with grad enabled just in case)
            seg_loss = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

        # 3. Total Loss
        total_loss = cls_loss + (self.seg_weight * seg_loss)

        return {"loss": total_loss, "cls_loss": cls_loss, "seg_loss": seg_loss}
