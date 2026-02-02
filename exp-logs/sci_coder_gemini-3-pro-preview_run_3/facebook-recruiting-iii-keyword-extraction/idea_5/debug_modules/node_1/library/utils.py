import os
import random
import numpy as np
import torch
import logging
import sys
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, level=logging.INFO):
    """
    Creates and configures a logger instance.

    Args:
        name (str): The name of the logger.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Calculates the Mean F1-Score (samples average) for multi-label classification.

    Args:
        outputs (torch.Tensor): Raw logits from the model [batch_size, num_classes].
        targets (torch.Tensor): Binary ground truth labels [batch_size, num_classes].
        threshold (float): Probability threshold for converting logits to binary predictions.

    Returns:
        float: The Mean F1-Score.
    """
    with torch.no_grad():
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(outputs)

        # Convert to binary predictions based on threshold
        preds = (probs > threshold).float()

        # Move to CPU and convert to numpy for sklearn
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate F1 score with 'samples' average
        # This calculates metrics for each instance, and finds their average
        score = f1_score(targets_np, preds_np, average="samples", zero_division=0)

    return score


def save_checkpoint(model, optimizer, epoch, metric, path):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        metric (float): Validation metric (e.g., F1 score) at this checkpoint.
        path (str): File path to save the checkpoint.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        tuple: (start_epoch, best_metric)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("metric", 0.0)

    return start_epoch, best_metric


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    Assumes a higher score is better (e.g., F1 Score).
    """

    def __init__(
        self,
        patience=Config.PATIENCE,
        delta=0,
        verbose=False,
        path=Config.MODEL_SAVE_PATH,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.inf

    def __call__(self, val_score, model, optimizer, epoch):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_score, model, optimizer, epoch):
        """Saves model when validation score increases."""
        if self.verbose:
            # Printing full precision as requested
            print(
                f"Validation score improved ({self.val_score_max} --> {val_score}).  Saving model ..."
            )
        save_checkpoint(model, optimizer, epoch, val_score, self.path)
        self.val_score_max = val_score
