import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.dataset import get_dataframe, TARGET_COLS

# Constants for caching
CACHE_DIR = "./working/idea_3/"
WEIGHTS_FILE = "class_weights.npy"


def get_class_weights(load_cached_data=True):
    """
    Calculates inverse class frequency weights based on the training dataset.
    Implements caching to avoid re-computing on every run.

    Args:
        load_cached_data (bool): If True, attempts to load weights from cache.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, WEIGHTS_FILE)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.from_numpy(weights_np).float()
        except Exception:
            # If loading fails, proceed to compute from scratch
            pass

    # 2. Compute from scratch
    # Load training metadata
    df = get_dataframe("train", load_cached_data=load_cached_data)

    # Calculate the sum of one-hot encoded labels to get class counts
    counts = df[TARGET_COLS].sum().values

    # Calculate Inverse Frequency Weights
    # Formula: Total_Samples / (Num_Classes * Class_Count)
    # This is the standard 'balanced' heuristic (e.g., sklearn)
    total_samples = counts.sum()
    n_classes = len(TARGET_COLS)

    # Add a small epsilon to safety-check against division by zero
    weights_np = total_samples / (n_classes * (counts + 1e-6))

    # 3. Save to cache
    try:
        np.save(cache_path, weights_np)
    except Exception as e:
        print(f"Warning: Failed to save class weights cache. Error: {e}")

    return torch.from_numpy(weights_np).float()


class WeightedSoftCrossEntropy(nn.Module):
    """
    A Cross Entropy Loss function that supports:
    1. Soft Targets (probabilities), required for CutMix/MixUp.
    2. Class Weighting, to handle dataset imbalance.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (torch.Tensor, optional): A tensor of class weights.
                                              Shape: (num_classes,).
        """
        super(WeightedSoftCrossEntropy, self).__init__()
        # Using register_buffer ensures 'weights' is part of the state_dict
        # and moves to the correct device (GPU/CPU) with the model.
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model predictions (logits) of shape (Batch, NumClasses).
            targets (torch.Tensor): Ground truth labels of shape (Batch, NumClasses).
                                    Can be one-hot encoded or soft probabilities.

        Returns:
            torch.Tensor: Scalar loss (mean over the batch).
        """
        # Calculate log probabilities from logits
        # log_softmax is numerically stable
        log_probs = F.log_softmax(logits, dim=1)

        # Standard Cross Entropy: - sum( target * log(prediction) )
        # We do this element-wise first
        loss = -targets * log_probs

        # Apply Class Weights if provided
        if self.weights is not None:
            # Broadcast weights (C,) to (B, C) and multiply
            loss = loss * self.weights

        # Sum over classes (dim=1) to get the total loss for each sample
        sample_losses = loss.sum(dim=1)

        # Return the mean loss over the batch
        return sample_losses.mean()
