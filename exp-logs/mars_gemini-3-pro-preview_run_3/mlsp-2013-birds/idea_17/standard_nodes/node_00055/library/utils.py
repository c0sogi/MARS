import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve (AUC) for multi-label classification.
    Robustly handles cases where specific classes are missing in the ground truth of the validation set.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, Num_Classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, Num_Classes).

    Returns:
        float: Macro-averaged AUC score. Returns 0.5 if no classes can be evaluated.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    auc_scores = []

    for i in range(num_classes):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        # ROC AUC is only defined if both classes (0 and 1) are present
        if len(np.unique(true_col)) > 1:
            try:
                score = roc_auc_score(true_col, pred_col)
                auc_scores.append(score)
            except ValueError:
                pass

    if not auc_scores:
        return 0.5

    return np.mean(auc_scores)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation metric stops improving.
    Saves the best model checkpoint.
    """

    def __init__(
        self, patience=7, mode="max", delta=0, verbose=False, path="checkpoint.pth"
    ):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            mode (str): 'max' for metrics like AUC, 'min' for metrics like Loss.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints messages on improvement.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.check_improvement = lambda current, best: current < best - delta
        else:
            self.check_improvement = lambda current, best: current > best + delta

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif not self.check_improvement(score, self.best_score):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """
        Saves the model state dict when the metric improves.
        """
        if self.verbose:
            print(f"Validation metric improved. Saving model to {self.path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)


def save_checkpoint(state, filename):
    """
    General utility to save a checkpoint dictionary (model, optimizer, scheduler, etc.).

    Args:
        state (dict): The state dictionary to save.
        filename (str): The full path where the checkpoint will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def write_submission(ids, probabilities, filename=None):
    """
    Generates the submission CSV file in the required format.

    Args:
        ids (list or np.ndarray): List of 'Id' strings/integers (e.g., 100, 101, ...).
        probabilities (list or np.ndarray): List of predicted probabilities.
        filename (str, optional): Path to save the submission file. Defaults to Config.SUBMISSION_FILE.
    """
    if filename is None:
        filename = Config.SUBMISSION_FILE

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame({"Id": ids, "Probability": probabilities})

    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def get_logger(log_file=None):
    """
    Creates a logger that writes to both a file and the console.

    Args:
        log_file (str, optional): Path to the log file. If None, logs only to console.

    Returns:
        logging.Logger: Configured logger.
    """
    import logging

    logger = logging.getLogger("BirdSpeciesLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # Console Handler
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger
