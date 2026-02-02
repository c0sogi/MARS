import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_robust_auc(y_true, y_pred):
    """
    Calculates the Macro-Average ROC AUC score, robust to missing classes in the batch.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: The mean ROC AUC score over classes present in y_true.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert lists to numpy if necessary
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    class_aucs = []
    n_classes = y_true.shape[1]

    for i in range(n_classes):
        # Only calculate AUC if the class has both positive and negative samples
        # np.unique returns sorted unique elements
        if len(np.unique(y_true[:, i])) == 2:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                class_aucs.append(auc)
            except ValueError:
                # Fallback for edge cases where roc_auc_score might still fail
                continue

    if not class_aucs:
        return 0.0

    return np.mean(class_aucs)


class CheckpointManager:
    """
    Manages the saving and deletion of model checkpoints, keeping only the top K best models.
    """

    def __init__(
        self,
        model_name,
        fold,
        save_dir=Config.CHECKPOINT_DIR,
        top_k=Config.TOP_K_CHECKPOINTS,
    ):
        """
        Args:
            model_name (str): Name of the model architecture.
            fold (int): The current fold number.
            save_dir (str): Directory to save checkpoints.
            top_k (int): Number of top checkpoints to keep.
        """
        self.model_name = model_name
        self.fold = fold
        self.save_dir = save_dir
        self.top_k = top_k
        self.checkpoints = []  # List of tuples: (score, filepath)

        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, model, metric, epoch):
        """
        Saves the model if the metric is among the top K seen so far.

        Args:
            model: The PyTorch model to save.
            metric (float): The validation metric (AUC) for this epoch.
            epoch (int): The current epoch number.

        Returns:
            bool: True if the model was saved, False otherwise.
        """
        # Define filename format
        filename = (
            f"{self.model_name}_fold_{self.fold}_epoch_{epoch}_auc_{metric:.5f}.pth"
        )
        filepath = os.path.join(self.save_dir, filename)

        should_save = False

        # If we haven't reached the limit, save it
        if len(self.checkpoints) < self.top_k:
            should_save = True
        else:
            # Check if current metric is better than the worst in our top_k list
            # List is sorted descending, so the last element is the worst
            min_score = self.checkpoints[-1][0]
            if metric > min_score:
                should_save = True

        if should_save:
            # Save the model state dict
            torch.save(model.state_dict(), filepath)

            # Add new checkpoint to the list
            self.checkpoints.append((metric, filepath))

            # Sort by score descending (higher AUC is better)
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Prune the list if it exceeds top_k
            if len(self.checkpoints) > self.top_k:
                to_remove = self.checkpoints.pop()  # Removes the last (worst) item
                remove_path = to_remove[1]
                if os.path.exists(remove_path):
                    try:
                        os.remove(remove_path)
                    except OSError:
                        pass  # Handle potential race conditions or permission issues gracefully

            return True

        return False
