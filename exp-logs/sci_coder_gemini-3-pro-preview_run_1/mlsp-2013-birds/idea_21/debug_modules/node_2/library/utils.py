import os
import random
import shutil
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Enforces deterministic CuDNN behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(log_file=None):
    """
    Creates a logger that writes to both console and a file.

    Args:
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def compute_roc_auc(y_true, y_pred):
    """
    Computes the macro-averaged ROC AUC score.
    Robustly handles cases where a class might be missing in the provided batch.

    Args:
        y_true (np.array): Ground truth labels (N, C)
        y_pred (np.array): Predicted probabilities (N, C)

    Returns:
        float: Macro ROC AUC score
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for NaNs in predictions and replace them
    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred)

    try:
        # Try computing standard macro AUC
        score = roc_auc_score(y_true, y_pred, average="macro")

        # Check if the result is NaN (can happen with degenerate data in debug mode)
        if np.isnan(score):
            raise ValueError("roc_auc_score returned NaN")
    except ValueError:
        # Fallback: Compute AUC per class and average, skipping classes with only one label present
        aucs = []
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            # Check if class exists in y_true (needs both 0 and 1 to compute AUC)
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    aucs.append(auc)
                except ValueError:
                    pass

        if len(aucs) == 0:
            score = 0.5  # Random guess if no classes can be evaluated
        else:
            score = np.mean(aucs)

    return score


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this is the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "model_best.pth"))


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        int: The epoch to resume from (start_epoch).
        float: The best metric score recorded.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    # Handle DataParallel wrapping if necessary (keys starting with 'module.')
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Load optimizer state
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score
