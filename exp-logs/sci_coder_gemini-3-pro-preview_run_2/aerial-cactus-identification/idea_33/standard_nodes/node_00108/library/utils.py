import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays or lists on CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def get_checkpoint_path(seed: int) -> str:
    """
    Constructs the file path for a model checkpoint based on the seed.

    Args:
        seed: The random seed used for the model instance.

    Returns:
        str: The full path to the checkpoint file.
    """
    filename = f"model_seed_{seed}.pth"
    return os.path.join(Config.WORKING_DIR, filename)


def save_checkpoint(state: dict, seed: int):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state: A dictionary containing model state, optimizer state, etc.
        seed: The random seed associated with this model instance.
    """
    path = get_checkpoint_path(seed)
    # Ensure directory exists (redundant if Config.setup() is called, but safe)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model: torch.nn.Module, seed: int, device: str = Config.DEVICE):
    """
    Loads the model weights from a checkpoint file.

    Args:
        model: The model architecture instance to load weights into.
        seed: The random seed identifying the checkpoint file.
        device: The device to map the location to (default: Config.DEVICE).

    Returns:
        The model with loaded weights.
    """
    path = get_checkpoint_path(seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    # Handle case where checkpoint is a dict containing 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Handle case where checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    return model


def save_submission(ids, predictions, path: str = Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids: List or array of image IDs (filenames).
        predictions: List or array of predicted probabilities.
        path: The file path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "has_cactus": predictions})

    df.to_csv(path, index=False)
