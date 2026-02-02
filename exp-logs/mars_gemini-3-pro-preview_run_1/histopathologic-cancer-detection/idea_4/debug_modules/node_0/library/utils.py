import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint to the specified file path.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        filepath (str): The full path where the checkpoint should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filepath)


class MetricTracker:
    """
    A utility class to track running averages of loss and calculate AUC.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the tracker."""
        self.val_loss_sum = 0.0
        self.count = 0
        self.targets = []
        self.predictions = []

    def update(self, loss, preds, targets):
        """
        Updates the metrics with a new batch of results.

        Args:
            loss (float): The average loss value for the batch (as returned by criterion).
            preds (torch.Tensor or np.array): Predicted probabilities.
            targets (torch.Tensor or np.array): Ground truth labels.
        """
        # Determine batch size to correctly weight the running average
        batch_size = len(targets)

        # Accumulate total loss (assuming input loss is mean per batch)
        self.val_loss_sum += loss * batch_size
        self.count += batch_size

        # Detach and move to CPU if tensors, then convert to list
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Flatten in case of extra dimensions and store
        self.predictions.extend(preds.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())

    def get_avg_loss(self):
        """
        Calculates the average loss over all updated samples.

        Returns:
            float: The average loss.
        """
        if self.count == 0:
            return 0.0
        return self.val_loss_sum / self.count

    def get_auc(self):
        """
        Calculates the Area Under the ROC Curve (AUC) for the accumulated data.

        Returns:
            float: The AUC score. Returns 0.5 if only one class is present or data is missing.
        """
        if len(self.targets) == 0:
            return 0.0

        # Check if both classes are present to avoid ValueError in roc_auc_score
        unique_classes = np.unique(self.targets)
        if len(unique_classes) < 2:
            return 0.5

        try:
            return roc_auc_score(self.targets, self.predictions)
        except ValueError:
            return 0.5
