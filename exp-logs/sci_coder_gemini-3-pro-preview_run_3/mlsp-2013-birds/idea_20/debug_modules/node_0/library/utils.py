import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Calculates the macro-averaged Area Under the ROC Curve (AUC).
    Robustly handles cases where a class is missing from the batch by skipping it.

    Args:
        y_true: Ground truth labels (N, num_classes). Can be numpy array or torch tensor.
        y_pred: Predicted probabilities (N, num_classes). Can be numpy array or torch tensor.

    Returns:
        float: The macro-averaged AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = y_true.shape[1]
    aucs = []

    for i in range(num_classes):
        # Only calculate AUC if the class has both positive and negative samples
        # np.unique returns sorted unique elements of an array
        if len(np.unique(y_true[:, i])) > 1:
            try:
                # roc_auc_score handles the calculation for a single class column
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # In case of any unexpected error in calculation, skip this class
                pass

    if len(aucs) == 0:
        return 0.0

    return np.mean(aucs)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """
    Simple logger that prints to console and writes to a file.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Initialize/Clear file
        with open(self.log_file, "w") as f:
            pass

    def log(self, message):
        """
        Prints message to console and appends to log file.
        """
        print(message)
        with open(self.log_file, "a") as f:
            f.write(str(message) + "\n")
