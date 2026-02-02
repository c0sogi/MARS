import os
import random
import numpy as np
import pandas as pd
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    """
    Calculates the Macro F1 score, which is the primary metric for this task.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels.
        y_pred (np.ndarray or torch.Tensor): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro F1
    # In "macro" F1 a separate F1 score is calculated for each species value and then averaged.
    score = f1_score(y_true, y_pred, average="macro")

    return score


def get_label_mappings(metadata_path):
    """
    Creates mappings between original category_ids and contiguous targets.

    Args:
        metadata_path (str): Path to the training metadata CSV.

    Returns:
        tuple: (id2label, label2id) dictionaries.
    """
    df = pd.read_csv(metadata_path)
    # Get unique category_ids sorted
    unique_ids = sorted(df["category_id"].unique())

    # Map original ID -> Contiguous Index (0 to N-1)
    id2label = {original_id: idx for idx, original_id in enumerate(unique_ids)}

    # Map Contiguous Index -> Original ID
    label2id = {idx: original_id for idx, original_id in enumerate(unique_ids)}

    return id2label, label2id
