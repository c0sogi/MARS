import os
import random
import numpy as np
import torch
import time
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Returns the PyTorch device (GPU if available, else CPU).

    Returns:
        torch.device: The computed device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def save_checkpoint(state: dict, filename: str):
    """
    Saves the model training state to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch or loss).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class EarlyStopping:
    """
    Implements Early Stopping to terminate training when validation loss stops improving.
    Also handles saving the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = Config.PATIENCE,
        verbose: bool = False,
        delta: float = 0,
        path: str = "checkpoint.pth",
    ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(
        self,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = 0,
    ):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ):
        """Saves model when validation loss decreases."""
        if self.verbose:
            # Printing full precision as requested
            print(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()

        save_checkpoint(state, self.path)
        self.val_loss_min = val_loss


def log_training_metrics(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float = None,
    time_elapsed: float = None,
):
    """
    Logs training metrics to console with full precision.

    Args:
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        train_loss (float): Training loss for the epoch.
        val_loss (float): Validation loss for the epoch.
        val_accuracy (float, optional): Validation accuracy.
        time_elapsed (float, optional): Time taken for the epoch in seconds.
    """
    msg = f"Epoch {epoch}/{total_epochs}"
    if time_elapsed is not None:
        msg += f" | Time: {time_elapsed}s"

    msg += f" | Train Loss: {train_loss}"
    msg += f" | Val Loss: {val_loss}"

    if val_accuracy is not None:
        msg += f" | Val Accuracy: {val_accuracy}"

    print(msg)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training loops.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
