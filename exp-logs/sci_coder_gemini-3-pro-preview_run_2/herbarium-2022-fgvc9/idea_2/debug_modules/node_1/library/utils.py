import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def calculate_macro_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score for the given ground truth and predictions.

    Args:
        y_true (array-like or torch.Tensor): Ground truth labels.
        y_pred (array-like or torch.Tensor): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro F1 score
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    # This does not take label imbalance into account, which matches the competition metric.
    return f1_score(y_true, y_pred, average="macro")


def save_checkpoint(model, optimizer, epoch, score, path=Config.MODEL_SAVE_PATH):
    """
    Saves the model checkpoint including model state, optimizer state, epoch, and score.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used.
        epoch (int): The current training epoch.
        score (float): The validation score (Macro F1) at this checkpoint.
        path (str): The file path to save the checkpoint. Defaults to Config.MODEL_SAVE_PATH.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Prepare the state dictionary
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": score,
    }

    # Save the checkpoint
    torch.save(state, path)
