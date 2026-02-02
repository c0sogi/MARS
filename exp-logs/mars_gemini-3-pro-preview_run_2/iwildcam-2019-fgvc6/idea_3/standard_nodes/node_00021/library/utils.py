import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_macro_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")


def compute_class_weights(df, target_col="Category"):
    """
    Computes class weights inversely proportional to class frequencies.

    Args:
        df (pd.DataFrame): The training metadata dataframe.
        target_col (str): The name of the target column containing class labels.

    Returns:
        torch.FloatTensor: A tensor of weights for each class, ordered by class index.
    """
    # Extract all labels
    y = df[target_col].values

    # Get unique classes present in the data
    classes = np.unique(y)

    # Compute weights using sklearn
    # class_weight = n_samples / (n_classes * np.bincount(y))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    # Create a weight array for all possible classes defined in Config
    # This handles cases where some classes might be missing from the specific df split provided
    # though ideally training set should have all classes.
    num_classes = Config.NUM_CLASSES
    final_weights = np.ones(num_classes, dtype=np.float32)

    # Map computed weights to their indices
    for cls, weight in zip(classes, weights):
        if cls < num_classes:
            final_weights[cls] = weight

    return torch.FloatTensor(final_weights).to(Config.DEVICE)
