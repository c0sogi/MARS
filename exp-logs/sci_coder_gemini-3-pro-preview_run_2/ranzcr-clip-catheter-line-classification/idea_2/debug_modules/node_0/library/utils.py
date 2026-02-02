import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC curve for each label and computes the average.
    Prints the individual AUCs for each column with full precision.

    Args:
        y_true: Ground truth labels. Can be a numpy array or torch.Tensor.
        y_pred: Predicted probabilities. Can be a numpy array or torch.Tensor.

    Returns:
        float: The average of the individual AUCs.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    target_cols = Config.TARGET_COLS
    aucs = []

    # Check for shape consistency
    if y_true.shape[1] != len(target_cols):
        print(
            f"Warning: y_true columns ({y_true.shape[1]}) do not match Config.TARGET_COLS ({len(target_cols)})"
        )

    for i, col_name in enumerate(target_cols):
        # roc_auc_score requires at least one positive and one negative sample
        if len(np.unique(y_true[:, i])) == 2:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(score)
            print(f"{col_name}: {score}")
        else:
            # If a class is missing in the validation set, we cannot compute AUC for it.
            # We exclude it from the average to avoid errors or artificial scores.
            print(f"{col_name}: Undefined (Only one class present in targets)")

    if len(aucs) > 0:
        avg_auc = np.mean(aucs)
    else:
        avg_auc = 0.0

    print(f"Average AUC: {avg_auc}")

    return avg_auc
