import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file):
    """
    Sets up a logger that writes to a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent double logging if attached to root

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)

    # Create formatters
    formatter = logging.Formatter("%(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def calculate_metric(y_pred_logits, y_true, threshold=0.5):
    """
    Calculates the Mean F1-Score (Macro) from raw logits and ground truth labels.

    Args:
        y_pred_logits (torch.Tensor or np.array): Raw model outputs (logits).
        y_true (torch.Tensor or np.array): Ground truth binary labels.
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        float: The macro-averaged F1 score.
    """
    # Convert tensors to numpy
    if isinstance(y_pred_logits, torch.Tensor):
        y_pred_logits = y_pred_logits.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Apply sigmoid to get probabilities
    y_pred_probs = 1.0 / (1.0 + np.exp(-y_pred_logits))

    # Binarize predictions
    y_pred_binary = (y_pred_probs > threshold).astype(int)

    # Calculate Macro F1 Score
    # zero_division=0 handles cases where a class is not predicted/present
    score = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)

    return score


class AverageMeter:
    """
    Computes and stores the average and current value.
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


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        filename (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=5,
        mode="max",
        delta=0.0,
        save_path="checkpoint.pth",
        verbose=False,
    ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            mode (str): 'min' for loss, 'max' for metric like F1.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model.
            verbose (bool): If True, prints messages.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.val_score_max = -np.inf

        if self.mode == "min":
            self.check_func = lambda current, best: current < best - self.delta
        else:
            self.check_func = lambda current, best: current > best + self.delta

    def __call__(self, score, model, optimizer=None, scheduler=None, epoch=None):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer, scheduler, epoch)
        else:
            if self.check_func(score, self.best_score):
                self.best_score = score
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model, optimizer, scheduler, epoch):
        """Saves model when validation metric improves."""
        if self.verbose:
            if self.mode == "min":
                print(
                    f"Validation metric decreased ({self.val_score_min} --> {score}).  Saving model ..."
                )
            else:
                print(
                    f"Validation metric increased ({self.val_score_max} --> {score}).  Saving model ..."
                )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "score": score,
        }
        save_checkpoint(state, self.save_path)

        if self.mode == "min":
            self.val_score_min = score
        else:
            self.val_score_max = score
