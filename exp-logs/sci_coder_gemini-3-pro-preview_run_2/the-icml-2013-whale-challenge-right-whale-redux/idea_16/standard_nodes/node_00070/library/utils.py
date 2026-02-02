import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels.
        y_pred (np.array or torch.Tensor): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate AUC
    # Note: roc_auc_score will raise a ValueError if y_true has only one class.
    # We allow this to propagate to alert the user of invalid validation splits.
    return roc_auc_score(y_true, y_pred)


class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric stops improving.
    Supports both maximization (e.g., AUC) and minimization (e.g., Loss).
    """

    def __init__(
        self,
        patience=Config.EARLY_STOPPING_PATIENCE,
        mode=Config.EARLY_STOPPING_MODE,
        delta=0.0,
        save_path=None,
    ):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            mode (str): One of {'min', 'max'}.
                        'min' for metrics like loss (improvement = decrease).
                        'max' for metrics like AUC (improvement = increase).
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.early_stop = False

        if self.mode == "min":
            self.best_score = np.inf
        elif self.mode == "max":
            self.best_score = -np.inf
        else:
            raise ValueError(
                f"EarlyStopping mode '{mode}' is unknown. Use 'min' or 'max'."
            )

    def __call__(self, score, model):
        """
        Updates the internal state based on the new score and saves the model if improved.

        Args:
            score (float): The current value of the monitored metric.
            model (torch.nn.Module): The model to save.
        """
        improved = False

        if self.mode == "max":
            if score > (self.best_score + self.delta):
                improved = True
        else:  # mode == 'min'
            if score < (self.best_score - self.delta):
                improved = True

        if improved:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Saves the model state dictionary to the specified path."""
        if self.save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            # Handle DataParallel wrappers if present
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), self.save_path)
            else:
                torch.save(model.state_dict(), self.save_path)
