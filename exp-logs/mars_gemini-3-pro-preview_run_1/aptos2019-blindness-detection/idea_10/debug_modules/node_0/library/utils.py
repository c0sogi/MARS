import numpy as np
import torch
from copy import deepcopy
from library.config import seed_everything


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric, which measures the agreement
    between two ratings. This metric typically varies from 0 (random agreement)
    to 1 (complete agreement).

    Args:
        y_true: Array-like of ground truth labels (integers 0-4).
        y_pred: Array-like of predicted scores. These will be rounded to the
                nearest integer and clipped to the range [0, 4].

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred)

    # Round continuous predictions and clip to valid range [0, 4]
    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, 0, 4)

    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    num_classes = 5

    # 1. Construct Observed Matrix O (Confusion Matrix)
    # O[i, j] corresponds to number of images with true rating i and predicted rating j
    O = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    # 2. Construct Weight Matrix w
    # Weights are calculated based on squared difference between scores
    w = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # 3. Construct Expected Matrix E
    # Calculated as outer product of histograms, assuming no correlation
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    total_samples = np.sum(O)

    # E must have the same sum as O
    E = np.outer(hist_true, hist_pred) / total_samples

    # 4. Calculate Kappa
    numerator = np.sum(w * O)
    denominator = np.sum(w * E)

    # Handle edge case where denominator is 0 (usually implies perfect agreement on a single class)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return 1.0 - (numerator / denominator)


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model weights.
    Maintains a shadow copy of the model that is updated as a weighted average
    of the training model's parameters. This often leads to better generalization
    and stability.
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for the moving average (default: 0.999).
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.shadow = deepcopy(model)

        # Detach parameters from the graph and set to eval mode
        for param in self.shadow.parameters():
            param.detach_()
        self.shadow.eval()

    def update(self, model):
        """
        Update the shadow model parameters.

        Args:
            model (torch.nn.Module): The current training model state.
        """
        with torch.no_grad():
            # Update parameters: shadow = decay * shadow + (1 - decay) * current
            for shadow_param, model_param in zip(
                self.shadow.parameters(), model.parameters()
            ):
                shadow_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1.0 - self.decay
                )

            # Copy buffers (e.g., BatchNorm running mean/var) directly from source
            for shadow_buffer, model_buffer in zip(
                self.shadow.buffers(), model.buffers()
            ):
                shadow_buffer.copy_(model_buffer)

    def get_model(self):
        """
        Returns the shadow (EMA) model.

        Returns:
            torch.nn.Module: The model with averaged weights.
        """
        return self.shadow
