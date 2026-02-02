import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred) -> float:
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: Ground truth binary labels (0 or 1). Can be a numpy array or torch Tensor.
        y_pred: Predicted probabilities for the positive class. Can be a numpy array or torch Tensor.

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(
    state: dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint.pth"
):
    """
    Saves the model state to a checkpoint file. If the model is the best so far,
    copies it to a separate 'model_best.pth' file.

    Args:
        state (dict): The state dictionary to save (e.g., model weights, optimizer, epoch).
        is_best (bool): True if this checkpoint has the best validation metric so far.
        checkpoint_dir (str): The directory where checkpoints should be saved.
        filename (str): The name of the checkpoint file. Default is 'checkpoint.pth'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, filename)

    # Save the checkpoint
    torch.save(state, file_path)

    # If this is the best model, create a copy
    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(file_path, best_path)
