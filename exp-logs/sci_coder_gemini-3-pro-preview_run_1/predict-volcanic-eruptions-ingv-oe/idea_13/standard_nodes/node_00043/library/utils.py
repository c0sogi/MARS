import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error
from library.config import Config


def seed_everything(seed: int = 42):
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


def metric_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.
    """
    return mean_absolute_error(y_true, y_pred)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint including optimizer state and current epoch.
    Ensures the parent directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint. Returns the epoch and loss recorded in the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss


def save_npy(data: np.ndarray, path: str):
    """
    Saves a numpy array to a .npy file.
    Ensures the parent directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_npy(path: str):
    """
    Loads a numpy array from a .npy file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found at {path}")
    return np.load(path)


def save_parquet(df: pd.DataFrame, path: str):
    """
    Saves a pandas DataFrame to a parquet file.
    Ensures the parent directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str):
    """
    Loads a pandas DataFrame from a parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found at {path}")
    return pd.read_parquet(path)


def transform_target(y):
    """
    Applies log1p transformation to the target variable.
    Used during training to normalize the target distribution.
    """
    return np.log1p(y)


def inverse_transform_target(y):
    """
    Applies expm1 transformation to the target variable.
    Used during inference to convert predictions back to the original scale.
    """
    # Cite debug_lesson_5: Handle NaNs and Infs explicitly to prevent downstream errors.
    # Replace NaN with 0 and Inf with finite bounds.
    y = np.nan_to_num(y, nan=0.0, posinf=32.0, neginf=-32.0)

    # Clip to a safe range to prevent overflow in expm1.
    # Max target is ~5e7, log(5e7) ~ 17.7.
    # 32.0 allows for predictions well above max target without overflowing float32.
    y = np.clip(y, -32.0, 32.0)

    return np.expm1(y)
