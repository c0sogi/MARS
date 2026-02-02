import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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

    # Ensure inputs are numpy arrays (in case they were lists)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint including model state, optimizer state, epoch, and loss.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.device):
    """
    Loads a model checkpoint. Updates the model and optimizer in-place.
    Returns the epoch and loss from the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
