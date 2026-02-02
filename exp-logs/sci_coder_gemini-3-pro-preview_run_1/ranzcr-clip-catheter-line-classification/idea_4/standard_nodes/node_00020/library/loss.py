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
    Loss function for Multi-Label Classification.
    Uses Weighted BCE With Logits.
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

    def forward(
        self,
        cls_logits,
        cls_targets,
        seg_logits=None,
        seg_targets=None,
        mask_validity=None,
    ):
        """
        Args:
            cls_logits: (B, NumClasses)
            cls_targets: (B, NumClasses)
            seg_logits: Ignored
            seg_targets: Ignored
            mask_validity: Ignored

        Returns:
            dict: Dictionary containing total loss.
        """
        # 1. Classification Loss
        cls_loss = self.cls_criterion(cls_logits, cls_targets)

        return {"loss": cls_loss}
