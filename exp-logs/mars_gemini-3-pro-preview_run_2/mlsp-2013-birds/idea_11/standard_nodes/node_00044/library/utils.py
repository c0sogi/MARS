import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across standard libraries, numpy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def calculate_auc(y_true, y_pred):
    """
    Calculates the Macro-Averaged Area Under the ROC Curve.
    Robustly handles cases where a class might be absent in the ground truth
    for a specific validation batch or fold by skipping those classes.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor of shape [N, num_classes])
        y_pred: Predicted probabilities (numpy array or torch tensor of shape [N, num_classes])

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.0 if no classes can be evaluated.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    aucs = []

    for i in range(num_classes):
        # Only calculate AUC if the class exists in the ground truth (has both 0 and 1, or at least one present if checking against pred)
        # Sklearn requires at least one positive and one negative to compute ROC AUC usually,
        # or at least the presence of the class.
        # Here we check if there is more than one unique value (i.e., both 0 and 1 are not strictly required by this check
        # but if only 0s exist, AUC is undefined/ill-defined for that specific target in isolation).
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # Fallback for edge cases where sklearn might still complain
                pass

    if not aucs:
        return 0.0

    return np.mean(aucs)


def save_checkpoint(model, path):
    """
    Saves the model's state dictionary to the specified path.
    Ensures the directory exists before saving.

    Args:
        model: PyTorch model instance.
        path (str): Destination path for the checkpoint file.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def compute_pos_weight(y_train):
    """
    Computes positive weights for BCEWithLogitsLoss based on class prevalence in the training data.
    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        y_train: Training labels (numpy array, pandas DataFrame, or torch Tensor).

    Returns:
        torch.Tensor: Tensor of weights with shape [num_classes] ready for BCEWithLogitsLoss.
    """
    # Handle pandas DataFrame
    if hasattr(y_train, "values"):
        y_train = y_train.values
    # Handle torch Tensor
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.detach().cpu().numpy()

    y_train = np.array(y_train)

    # Calculate counts
    num_positives = y_train.sum(axis=0)
    num_negatives = y_train.shape[0] - num_positives

    # Clip positives to avoid division by zero (though unlikely in full dataset)
    num_positives = np.clip(num_positives, a_min=1, a_max=None)

    weights = num_negatives / num_positives

    return torch.tensor(weights, dtype=torch.float32)
