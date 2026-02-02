import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import CFG


def seed_everything(seed=CFG.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to CFG.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        # This is equivalent to Mean Column-wise ROC AUC.
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This can happen if a class is not present in the batch (only one unique value).
        # We fall back to manual calculation per column, skipping undefined ones.
        scores = []
        for i in range(y_true.shape[1]):
            try:
                # Check if class has both 0 and 1
                if len(np.unique(y_true[:, i])) > 1:
                    score = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(score)
            except ValueError:
                pass

        if not scores:
            return 0.0
        return np.mean(scores)


def get_class_weights(df, target_cols):
    """
    Calculates inverse frequency weights (pos_weight) for use with BCEWithLogitsLoss.
    Formula: weight = negative_count / positive_count

    Args:
        df (pd.DataFrame): The dataframe containing the target labels.
        target_cols (list): List of column names corresponding to the targets.

    Returns:
        torch.Tensor: A tensor of shape (len(target_cols),) containing the weights.
    """
    weights = []
    for col in target_cols:
        pos_count = df[col].sum()
        total_count = len(df)
        neg_count = total_count - pos_count

        # Prevent division by zero if a class has no positive samples (unlikely in training)
        if pos_count == 0:
            weight = 1.0
        else:
            weight = neg_count / pos_count

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
