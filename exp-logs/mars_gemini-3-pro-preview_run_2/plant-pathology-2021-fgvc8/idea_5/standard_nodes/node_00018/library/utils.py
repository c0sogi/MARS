import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic cuDNN algorithms.

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


def calculate_f1_score(y_true, y_pred, threshold=0.5, average="macro"):
    """
    Calculates the Mean F1-Score for multi-label classification.
    Handles conversion from logits/probabilities to binary predictions.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (binary).
        y_pred (np.array or torch.Tensor): Predictions (logits or probabilities).
        threshold (float): Threshold for converting probabilities to binary labels.
        average (str): Averaging method for F1 score ('macro' is standard for Mean F1).

    Returns:
        float: The calculated F1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if y_pred appears to be logits (values outside [0, 1])
    # If so, apply sigmoid
    if y_pred.size > 0:
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = 1 / (1 + np.exp(-y_pred))

    # Binarize predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true = y_true.astype(int)

    # Calculate F1 score
    # Zero_division=0 prevents warnings for classes with no predicted samples
    score = f1_score(y_true, y_pred_binary, average=average, zero_division=0)
    return score


def save_checkpoint(model, optimizer, epoch, score, filename="model_best.pth"):
    """
    Saves the model checkpoint. Uses deepcopy to ensure the saved state_dict
    is an immutable snapshot of the model at this specific point in time.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer (optional).
        epoch (int): Current epoch.
        score (float): Validation score.
        filename (str): Filename to save in Config.WORKING_DIR.
    """
    save_path = os.path.join(Config.WORKING_DIR, filename)

    # Deepcopy the state dict to prevent reference issues
    model_state_dict = copy.deepcopy(model.state_dict())

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "score": score,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path} with score: {score}")
