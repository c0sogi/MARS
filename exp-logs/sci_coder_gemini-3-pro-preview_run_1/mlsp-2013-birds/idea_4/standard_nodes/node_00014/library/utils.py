import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state(model, path):
    """
    Saves a deep copy of the model's state dictionary to the specified path.
    This ensures that the saved state is an immutable snapshot of the model
    at the time of saving, preventing reference issues during training.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the state dict will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a deep copy of the state dictionary
    state_dict_snapshot = copy.deepcopy(model.state_dict())

    # Save the snapshot
    torch.save(state_dict_snapshot, path)


def calculate_metric(y_true, y_pred):
    """
    Calculates the Macro-Averaged ROC AUC score for multi-label classification.
    Handles cases where specific classes may not be present in the ground truth
    (e.g., in a small validation batch) by skipping those columns.

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels (N, NumClasses).
        y_pred (np.array or torch.Tensor): Predicted probabilities (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.0 if no valid classes are found.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    auc_scores = []

    for i in range(num_classes):
        # We can only calculate ROC AUC if there is at least one positive
        # and one negative sample for the class in the provided set.
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # Skip classes that cause calculation errors
                pass

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)
