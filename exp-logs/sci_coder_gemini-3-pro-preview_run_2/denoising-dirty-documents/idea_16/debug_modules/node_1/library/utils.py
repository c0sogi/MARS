import os
import random
import numpy as np
import torch
from library.config import CHECKPOINT_DIR, DEVICE


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both numpy arrays and torch tensors.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    """
    Saves the model and optimizer state to the configured checkpoint directory.
    """
    # Ensure the directory exists (redundant if config handles it, but safe)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    filepath = os.path.join(CHECKPOINT_DIR, filename)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, filepath)


def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    """
    Loads the model and optimizer state from a checkpoint file.
    Returns the epoch and loss recorded in the checkpoint.
    If file does not exist, returns None.
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(filepath):
        return None

    checkpoint = torch.load(filepath, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
