import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path):
    """
    Saves the model state dictionary to the specified path.
    Ensures the directory exists before saving.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model to.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Handle DataParallel or DistributedDataParallel wrappers if present
    if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, path)


def load_model(model, path, device="cpu"):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path to the saved model weights.
        device (str or torch.device): The device to map the location to.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def sanitize_pseudo_labels(pseudo_labels):
    """
    Checks for NaN values in the generated pseudo-labels and ensures data integrity.
    Explicitly asserts that merged targets contain no NaN values.

    Args:
        pseudo_labels (np.ndarray or torch.Tensor): The generated pseudo-labels (probabilities).

    Returns:
        np.ndarray: The sanitized pseudo-labels if valid.

    Raises:
        AssertionError: If NaN values are detected.
    """
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.detach().cpu().numpy()

    if np.isnan(pseudo_labels).any():
        nan_count = np.isnan(pseudo_labels).sum()
        total_count = pseudo_labels.size
        raise AssertionError(
            f"Pseudo-labels contain {nan_count}/{total_count} NaN values. "
            "Sanitization failed as strict integrity is required."
        )

    return pseudo_labels


def save_submission(ids, probabilities, output_path):
    """
    Saves predictions to a CSV file in the required submission format.

    Args:
        ids (list or np.ndarray): List of combined IDs.
        probabilities (list or np.ndarray): List of predicted probabilities.
        output_path (str): Path to save the CSV.
    """
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame({"Id": ids, "Probability": probabilities})

    df.to_csv(output_path, index=False)
