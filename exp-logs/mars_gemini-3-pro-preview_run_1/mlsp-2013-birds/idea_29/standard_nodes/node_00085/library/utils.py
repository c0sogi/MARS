import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(model, optimizer, epoch, filename):
    """
    Saves the model checkpoint to the specified filename.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save (can be None).
        epoch (int): The current epoch number.
        filename (str): Path to save the checkpoint file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(state, filename)


def load_checkpoint(model, filename, device=Config.DEVICE, optimizer=None):
    """
    Loads the model checkpoint from the specified filename.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        device (str): Device to map the location to.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.

    Returns:
        int: The epoch number stored in the checkpoint.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        # Filter out SWA-specific buffer 'n_averaged'
        if k == "n_averaged":
            continue

        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("epoch", 0)


def compute_roc_auc(y_true, y_pred):
    """
    Computes the macro-averaged ROC AUC score.

    Args:
        y_true: numpy array of shape (N, num_classes) containing ground truth (0 or 1).
        y_pred: numpy array of shape (N, num_classes) containing predicted probabilities.

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Check for valid inputs
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # roc_auc_score throws a ValueError if a class has only one label (all 0s or all 1s).
    # We attempt the standard macro calculation first. If it fails, we calculate per-class
    # and average only the valid classes.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                # Calculate AUC for this specific class
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # This class has only one label (all 0s or all 1s) in the provided batch
                pass

        if len(aucs) == 0:
            return 0.0
        score = np.mean(aucs)

    return score
