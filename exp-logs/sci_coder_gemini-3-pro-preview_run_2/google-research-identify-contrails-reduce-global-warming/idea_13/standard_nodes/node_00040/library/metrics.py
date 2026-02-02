import torch
import numpy as np


class GlobalDiceMetric:
    """
    Computes the Global Dice Coefficient.

    The metric is defined as: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of all predicted pixels across the entire dataset,
    and Y is the set of all ground truth pixels.

    This class accumulates statistics (intersection and union areas) over batches
    and computes the final score at the end of the epoch.
    """

    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-6):
        """
        Args:
            threshold (float): Threshold to convert probabilities to binary masks.
            epsilon (float): Small constant to avoid division by zero.
        """
        self.threshold = threshold
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        """
        Resets the internal state of the metric.
        Should be called at the start of each epoch.
        """
        self.intersection_sum = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates the metric state with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Model outputs (logits) of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W).
        """
        # Ensure we are working with detached tensors to avoid memory leaks
        preds = preds.detach()
        targets = targets.detach()

        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(preds)

        # Apply threshold to get binary predictions
        pred_mask = (probs > self.threshold).float()

        # Ensure targets are float for calculation
        target_mask = targets.float()

        # Flatten tensors to calculate global stats for this batch
        # We flatten everything to 1D array
        pred_flat = pred_mask.view(-1)
        target_flat = target_mask.view(-1)

        # Calculate intersection and sums for this batch
        intersection = (pred_flat * target_flat).sum().item()
        p_sum = pred_flat.sum().item()
        t_sum = target_flat.sum().item()

        # Accumulate
        self.intersection_sum += intersection
        self.pred_sum += p_sum
        self.target_sum += t_sum
        self.count += 1

    def compute(self) -> float:
        """
        Computes the final Global Dice Coefficient based on accumulated stats.

        Returns:
            float: The global Dice score.
        """
        denominator = self.pred_sum + self.target_sum

        # Handle edge case where both prediction and target are empty
        if denominator == 0:
            return (
                0.0 if self.intersection_sum == 0 else 1.0
            )  # Technically if both empty, dice is 1, but usually 0/0 handled via epsilon

        dice = (2.0 * self.intersection_sum) / (denominator + self.epsilon)
        return dice
