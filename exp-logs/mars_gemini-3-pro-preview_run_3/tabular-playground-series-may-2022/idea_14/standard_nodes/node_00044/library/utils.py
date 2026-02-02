import sys
import os
from sklearn.metrics import roc_auc_score

# Import set_seed from the provided library.config to ensure consistent reproducibility
# and avoid re-implementation as per instructions.
try:
    from library.config import set_seed
except ImportError:
    # Fallback in case library is not directly importable, though it should be
    # given the file structure provided in the problem description.
    import random
    import numpy as np
    import torch

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_score):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_score (array-like): Target scores (probability estimates of the positive class).

    Returns:
        float: The computed AUC score.
    """
    return roc_auc_score(y_true, y_score)
