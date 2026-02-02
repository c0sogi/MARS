import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed: int = Config.seed) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculates the Mean F1-Score (Macro F1) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: The Macro F1-Score.
    """
    # Convert probabilities to binary predictions based on threshold
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calculate Macro F1 score (average of F1 scores per class)
    # zero_division=0 ensures no errors if a class is not present in the batch
    score = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)

    return score


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Saves the model state dictionary to the specified path.
    Uses copy.deepcopy to ensure the saved weights are an immutable snapshot of the current state.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The file path where the model should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a deep copy of the state dictionary
    # This prevents subsequent in-place modifications by the optimizer (e.g., momentum buffers)
    # from affecting the saved weights in memory before they are written to disk.
    state_dict = copy.deepcopy(model.state_dict())

    # Save the state dictionary
    torch.save(state_dict, path)
