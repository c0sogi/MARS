import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The calculated AUC score.
    """
    return roc_auc_score(y_true, y_pred)


class Logger:
    """
    A simple logger that writes messages to both the console and a log file.
    """

    def __init__(self, file_path):
        """
        Initializes the Logger.

        Args:
            file_path (str): The path to the log file.
        """
        self.file_path = file_path

        # Ensure the directory exists
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Initialize the file (clear previous content)
        with open(self.file_path, "w") as f:
            pass

    def log(self, message):
        """
        Logs a message to the console and the file.

        Args:
            message (str): The message to log.
        """
        print(message)
        with open(self.file_path, "a") as f:
            f.write(str(message) + "\n")


def save_checkpoint(state, filename):
    """
    Saves a model checkpoint to a file.

    Args:
        state (dict): The state dictionary to save (e.g., model weights, optimizer state).
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure directory exists
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, device=None):
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): The path to the checkpoint file.
        device (torch.device, optional): The device to load the tensors onto.
                                        Defaults to CUDA if available, else CPU.

    Returns:
        dict: The loaded state dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.load(filename, map_location=device)
