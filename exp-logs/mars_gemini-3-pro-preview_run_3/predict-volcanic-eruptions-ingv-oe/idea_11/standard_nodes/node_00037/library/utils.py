import os
import random
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_absolute_error


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path: str):
    """
    Saves a model to the specified path using joblib.
    Creates the directory if it does not exist.

    Args:
        model: The model object to save.
        path (str): The file path where the model should be saved.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """
    Loads a model from the specified path using joblib.

    Args:
        path (str): The file path to load the model from.

    Returns:
        The loaded model object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values.

    Returns:
        float: The MAE score.
    """
    return mean_absolute_error(y_true, y_pred)
