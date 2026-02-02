import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations if available.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def save_model(model, path):
    """
    Saves the model state dictionary to the specified path.
    Creates the parent directory if it does not exist.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the state dictionary.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device=Config.DEVICE):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path to the saved state dictionary.
        device (str): The device to map the location to (e.g., 'cpu', 'cuda').

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def compute_auc(y_true, y_pred):
    """
    Computes the Macro-Averaged ROC AUC score for multi-label classification.
    Robustly handles cases where a class might be absent (all 0s) or ubiquitous (all 1s)
    in the provided batch/subset by excluding those classes from the average.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged AUC score. Returns 0.0 if no classes are valid.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Only compute AUC if there are both positive and negative samples for the class
        # sklearn's roc_auc_score throws an error if only one class is present.
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(score)
            except ValueError:
                # Fallback for any unexpected sklearn errors
                pass

    if len(aucs) == 0:
        return 0.0

    return np.mean(aucs)
