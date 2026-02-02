import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config


def get_class_weights(df, load_cached_data=True, device=Config.DEVICE):
    """
    Calculates class weights inversely proportional to class frequencies.
    Implements caching mechanism using .npy format.

    Args:
        df (pd.DataFrame): The training metadata dataframe containing 'stratify_label'.
        load_cached_data (bool): Whether to try loading from cache.
        device (torch.device): The device to move the weights tensor to.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    weights_np = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            # print(f"Loaded class weights from {cache_path}")
        except Exception as e:
            # print(f"Failed to load cache: {e}. Recalculating.")
            weights_np = None

    # 2. Calculate if not loaded
    if weights_np is None:
        # Check if stratify_label exists
        if "stratify_label" not in df.columns:
            # Fallback: try to derive from one-hot columns if stratify_label is missing
            # Assuming Config.CLASS_LABELS matches columns
            labels = []
            for _, row in df.iterrows():
                for cls in Config.CLASS_LABELS:
                    if row.get(cls, 0) == 1:
                        labels.append(cls)
                        break
                else:
                    labels.append(Config.CLASS_LABELS[0])  # Default fallback
        else:
            labels = df["stratify_label"].values

        # Count frequencies
        # Ensure we count in the order of Config.CLASS_LABELS
        class_counts = {cls: 0 for cls in Config.CLASS_LABELS}
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            if u in class_counts:
                class_counts[u] = c

        total_samples = sum(class_counts.values())
        num_classes = len(Config.CLASS_LABELS)

        # Calculate weights: Total / (Num_Classes * Class_Count)
        weights_list = []
        for cls in Config.CLASS_LABELS:
            count = class_counts[cls]
            if count > 0:
                w = total_samples / (num_classes * count)
            else:
                w = 1.0  # Fallback for empty classes
            weights_list.append(w)

        weights_np = np.array(weights_list, dtype=np.float32)

        # Save to cache
        np.save(cache_path, weights_np)
        # print(f"Calculated and saved class weights to {cache_path}")

    # 3. Convert to Tensor and return
    weights_tensor = torch.tensor(weights_np, dtype=torch.float32).to(device)
    return weights_tensor


class WeightedCrossEntropyLoss(nn.Module):
    """
    A wrapper around nn.CrossEntropyLoss that applies pre-calculated class weights.
    """

    def __init__(self, weights=None, device=Config.DEVICE):
        """
        Args:
            weights (torch.Tensor, optional): Class weights tensor.
            device (torch.device): Device where the loss computation happens.
        """
        super(WeightedCrossEntropyLoss, self).__init__()

        # If weights are provided, ensure they are on the correct device
        if weights is not None:
            weights = weights.to(device)

        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions (logits) of shape (Batch, Num_Classes).
            targets (torch.Tensor): Ground truth indices of shape (Batch,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(inputs, targets)
