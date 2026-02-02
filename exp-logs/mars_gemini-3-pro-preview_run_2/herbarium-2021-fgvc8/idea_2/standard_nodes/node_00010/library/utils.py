import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true: Ground truth labels (array-like or tensor).
        y_pred: Predicted labels (array-like or tensor).

    Returns:
        float: Macro F1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return f1_score(y_true, y_pred, average="macro")


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


def save_checkpoint(state, filename=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        filename (str): Path to save the checkpoint. If None, uses Config.MODEL_CHECKPOINT_PATH.
    """
    if filename is None:
        filename = Config.MODEL_CHECKPOINT_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        model: The model to load weights into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        filename (str): Path to the checkpoint file.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if filename is None:
        filename = Config.MODEL_CHECKPOINT_PATH

    if not os.path.isfile(filename):
        return None

    if device is None:
        device = Config.DEVICE

    checkpoint = torch.load(filename, map_location=device)

    # Handle both full checkpoint dicts and direct state dicts
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(self, patience=3, delta=0, path=None, verbose=False, mode="max"):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            verbose (bool): If True, prints a message for each validation improvement.
            mode (str): 'max' for metrics like F1 (higher is better), 'min' for loss (lower is better).
        """
        self.patience = patience
        self.delta = delta
        self.path = path if path is not None else Config.MODEL_CHECKPOINT_PATH
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

        if self.mode == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def __call__(self, score, model, optimizer=None, scheduler=None, epoch=None):
        if self.mode == "min":
            # Lower is better
            if score < self.best_score - self.delta:
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            # Higher is better
            if score > self.best_score + self.delta:
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
                self.best_score = score
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
        """Saves model when validation score improves."""
        if self.verbose:
            print(f"Validation score improved to {score:.6f}. Saving model...")

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "score": score,
        }
        if optimizer:
            state["optimizer"] = optimizer.state_dict()
        if scheduler:
            state["scheduler"] = scheduler.state_dict()

        save_checkpoint(state, filename=self.path)
