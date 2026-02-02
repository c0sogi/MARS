import os
import random
import numpy as np
import torch
from library.config import DEVICE, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Configures CuDNN for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device (CPU or CUDA) defined in config.
    """
    return torch.device(DEVICE)


def calculate_accuracy(outputs, targets):
    """
    Calculates multiclass accuracy.

    Args:
        outputs (torch.Tensor): Logits or probabilities of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth indices of shape (batch_size).

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)

        if total == 0:
            return 0.0

        return correct / total


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler.
        epoch (int): Current epoch.
        val_loss (float): Validation loss.
        path (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=DEVICE):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (containing epoch, loss, etc.).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, delta=0, path="checkpoint.pth", verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                # Printing full precision as requested, no rounding
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # Printing full precision as requested
            print(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, self.path)
        self.val_loss_min = val_loss
