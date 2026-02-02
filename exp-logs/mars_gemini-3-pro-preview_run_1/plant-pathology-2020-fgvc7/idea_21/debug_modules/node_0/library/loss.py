import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


def get_class_weights(
    df: pd.DataFrame, class_cols: list, device: str = "cpu"
) -> torch.Tensor:
    """
    Calculates inverse class frequency weights based on the provided dataframe.

    Formula: Weight_c = Total_Samples / (Num_Classes * Count_c)

    Args:
        df (pd.DataFrame): The metadata dataframe containing target labels.
        class_cols (list): List of column names corresponding to the target classes.
        device (str): The device to store the tensor on.

    Returns:
        torch.Tensor: A tensor of shape (Num_Classes,) containing the weights.
    """
    # Calculate the total mass (sum of soft labels or count of hard labels) for each class
    # We use sum() because labels might be soft probabilities or one-hot
    class_counts = df[class_cols].sum(axis=0).values

    total_samples = class_counts.sum()
    n_classes = len(class_cols)

    # Compute inverse frequency weights
    # Add a small epsilon to counts to prevent division by zero in pathological cases
    weights = total_samples / (n_classes * (class_counts + 1e-6))

    return torch.tensor(weights, dtype=torch.float32, device=device)


class WeightedSoftCrossEntropy(nn.Module):
    """
    Weighted Cross Entropy Loss for Soft Targets.

    Computes the cross entropy between logits and soft targets, applies class weights,
    and normalizes by the sum of weights in the batch (not batch size).
    """

    def __init__(self, class_weights: torch.Tensor = None):
        """
        Args:
            class_weights (torch.Tensor, optional): A tensor of shape (C,) containing
                                                    weights for each class.
        """
        super(WeightedSoftCrossEntropy, self).__init__()

        # Register weights as a buffer so they are part of the state_dict
        # and move to the correct device automatically with the model.
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model predictions of shape (Batch, Num_Classes).
                                   Expects raw logits (before Softmax).
            targets (torch.Tensor): Soft targets of shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute Log Softmax
        log_probs = F.log_softmax(logits, dim=1)

        # Compute element-wise Cross Entropy: - target * log(prediction)
        loss_per_element = -targets * log_probs

        # Apply Class Weights if provided
        if self.class_weights is not None:
            # View weights as (1, C) for broadcasting
            weights = self.class_weights.view(1, -1)

            # Weighted loss per element
            weighted_loss = loss_per_element * weights

            # Calculate normalization factor: Sum of weights in the batch
            # This is the sum of (target_prob * class_weight) for all samples and classes
            normalization_factor = (targets * weights).sum()
        else:
            # If no weights, standard soft cross entropy
            weighted_loss = loss_per_element

            # Normalization factor is just the sum of targets (usually equals Batch Size)
            normalization_factor = targets.sum()

        # Compute final loss
        # Sum of weighted losses divided by the sum of weights
        # We add a small epsilon to denominator for numerical stability
        return weighted_loss.sum() / (normalization_factor + 1e-8)
