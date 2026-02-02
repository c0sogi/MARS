import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.array): Ground truth labels (N_samples, N_classes).
        y_pred (np.array): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The mean ROC AUC score across all columns.
    """
    try:
        # average='macro' calculates the metric for each label, and finds their unweighted mean.
        # This matches "Mean column-wise ROC AUC".
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError as e:
        # Handle cases where a class might only have one label in the batch/fold
        print(f"Warning: Error calculating ROC AUC: {e}")
        return 0.0


def save_model(model, path):
    """
    Saves the PyTorch model's state dictionary to the specified path.
    Ensures the directory exists before saving.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The full file path for the saved model.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)
    # Print confirmation is usually helpful, but we'll keep it silent as per general strict instructions
    # unless specifically asked to print. The prompt says "Only print the required information",
    # so we will rely on the training loop to print "Saved model..." if needed.


def load_model(model, path, device):
    """
    Loads the PyTorch model's state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model architecture instance.
        path (str): The full file path to the saved weights.
        device (str): The device to load the weights onto ('cpu' or 'cuda').

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
