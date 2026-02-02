import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Macro-Averaged Area Under the ROC Curve.

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels (N, NumClasses).
        y_pred (np.array or torch.Tensor): Predicted probabilities (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle edge cases where a class might not be present in the validation set
    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        # This does not take label imbalance into account.
        score = roc_auc_score(y_true, y_pred, average="macro")
        if np.isnan(score):
            raise ValueError("ROC AUC score is NaN")
    except ValueError:
        # Fallback for batches/sets where some classes have only one unique label (all 0 or all 1)
        n_classes = y_true.shape[1]
        scores = []
        for i in range(n_classes):
            try:
                # Calculate AUC for this specific class
                s = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(s)
            except ValueError:
                # If only one class is present in y_true, AUC is undefined.
                # We treat this neutrally (0.5) to avoid crashing, though
                # in a proper validation set this shouldn't happen often.
                scores.append(0.5)
        score = np.mean(scores)

    return score


def save_checkpoint(model, path):
    """
    Safely saves the model state dictionary using deepcopy to ensure immutability.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The destination file path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Use deepcopy to avoid reference issues if the model is modified later
    state_dict = copy.deepcopy(model.state_dict())

    torch.save(state_dict, path)


class Logger:
    """
    A simple logger to track training progress and metrics.
    """

    def __init__(self, log_file=None):
        """
        Initialize the logger.

        Args:
            log_file (str, optional): Path to a file where logs should be written.
        """
        self.log_file = log_file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # Initialize/Clear the log file
            with open(self.log_file, "w") as f:
                f.write("")

    def log(self, message):
        """
        Logs a message to stdout and the log file.

        Args:
            message (str): The message to log.
        """
        print(message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_metrics(self, epoch, train_loss, val_loss, val_auc):
        """
        Logs training metrics with full precision.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
            val_auc (float): Validation ROC AUC.
        """
        # Printing full precision without rounding as requested
        message = f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val AUC = {val_auc}"
        self.log(message)
