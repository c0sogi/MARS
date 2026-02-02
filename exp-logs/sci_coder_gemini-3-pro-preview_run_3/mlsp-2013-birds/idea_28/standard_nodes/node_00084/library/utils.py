import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RobustMetric:
    """
    Calculates Area Under ROC Curve while handling missing classes in batches
    to prevent NaN errors. Accumulates predictions over an epoch or evaluation phase.
    """

    def __init__(self):
        self.predictions = []
        self.targets = []

    def update(self, outputs, targets):
        """
        Update the metric with a new batch of predictions and targets.

        Args:
            outputs: Model outputs (logits or probabilities), expected shape (batch_size, num_classes).
            targets: Ground truth labels, expected shape (batch_size, num_classes).
        """
        # Detach and move to CPU, convert to numpy
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.append(outputs)
        self.targets.append(targets)

    def compute(self):
        """
        Compute the mean AUROC across all classes.
        Handles cases where a class might not be present in the accumulated data.

        Returns:
            float: The mean AUROC score.
        """
        if not self.predictions:
            return 0.0

        # Concatenate all accumulated batches
        y_pred = np.vstack(self.predictions)
        y_true = np.vstack(self.targets)

        num_classes = y_true.shape[1]
        auc_scores = []

        for i in range(num_classes):
            # Only calculate AUC if the class has both positive and negative samples
            # This prevents ValueError from roc_auc_score
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    # roc_auc_score handles ranking, so logits are acceptable
                    score = roc_auc_score(y_true[:, i], y_pred[:, i])
                    auc_scores.append(score)
                except ValueError:
                    pass

        if not auc_scores:
            return 0.0

        return np.mean(auc_scores)

    def reset(self):
        """Reset the internal state."""
        self.predictions = []
        self.targets = []


class SnapshotManager:
    """
    Tracks and persists only the top-K best model checkpoints based on validation scores.
    Manages checkpoints independently per fold to support ensemble averaging.
    """

    def __init__(self, checkpoint_dir, k=Config.SNAPSHOTS_K, maximize=True):
        """
        Args:
            checkpoint_dir (str): Directory to save checkpoints.
            k (int): Number of top checkpoints to keep per fold.
            maximize (bool): Whether higher score is better (True for AUC).
        """
        self.checkpoint_dir = checkpoint_dir
        self.k = k
        self.maximize = maximize
        # Dictionary to track snapshots per fold: {fold_idx: [(score, path), ...]}
        self.snapshots = {}

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, score, epoch, fold_idx, model_name):
        """
        Potentially save the model if it's among the top-K for the given fold.
        Deletes the worst checkpoint if the limit K is exceeded.

        Args:
            model: The PyTorch model to save.
            score (float): The validation metric score.
            epoch (int): Current epoch number.
            fold_idx (int): Current fold index.
            model_name (str): Name of the model architecture.
        """
        if fold_idx not in self.snapshots:
            self.snapshots[fold_idx] = []

        fold_snapshots = self.snapshots[fold_idx]

        # Filename includes fold, epoch, and high precision score to avoid collisions
        filename = f"{model_name}_fold{fold_idx}_epoch{epoch}_auc{score:.6f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Determine if we should save
        should_save = False
        if len(fold_snapshots) < self.k:
            should_save = True
        else:
            # Sort current snapshots to find the worst one
            # If maximize=True, sort descending, so the last element is the worst
            fold_snapshots.sort(key=lambda x: x[0], reverse=self.maximize)
            worst_score = fold_snapshots[-1][0]

            if self.maximize:
                if score > worst_score:
                    should_save = True
            else:
                if score < worst_score:
                    should_save = True

        if should_save:
            # Save the new model
            torch.save(model.state_dict(), filepath)
            fold_snapshots.append((score, filepath))

            # Prune the list to keep only top K
            # Sort again to ensure correct order
            fold_snapshots.sort(key=lambda x: x[0], reverse=self.maximize)

            while len(fold_snapshots) > self.k:
                # Remove the worst model (last element)
                worst_entry = fold_snapshots.pop()
                worst_path = worst_entry[1]
                if os.path.exists(worst_path):
                    os.remove(worst_path)

            # Update the dictionary
            self.snapshots[fold_idx] = fold_snapshots
