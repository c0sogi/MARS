import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def compute_robust_auc(y_true, y_pred):
    """
    Calculates the Macro Averaged ROC AUC score, explicitly handling cases where
    specific classes might be absent (only one unique label) in the provided batch.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor), shape (N, num_classes).
        y_pred: Predicted probabilities (numpy array or torch tensor), shape (N, num_classes).

    Returns:
        float: The macro-averaged AUC score. Returns 0.5 if no classes can be evaluated.
    """
    # Convert tensors to numpy if passed as tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    valid_aucs = []

    for i in range(num_classes):
        # Extract columns for the current class
        class_true = y_true[:, i]
        class_pred = y_pred[:, i]

        # We can only calculate ROC AUC if there are both positive and negative samples
        if len(np.unique(class_true)) > 1:
            try:
                auc = roc_auc_score(class_true, class_pred)
                valid_aucs.append(auc)
            except ValueError:
                # Fallback for unexpected sklearn errors
                continue

    if not valid_aucs:
        # If no classes could be evaluated (e.g., extremely small batch with constant labels)
        return 0.5

    return np.mean(valid_aucs)


def save_checkpoint(state, filename):
    """
    Saves the model training state to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, epoch, etc.
        filename (str): Path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, filename, optimizer=None, device=Config.DEVICE):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the checkpoint location to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
