import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import torchaudio.functional as F
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_deltas(spec_tensor, win_length=5):
    """
    Computes the first and second temporal derivatives (delta and delta-delta)
    of a spectrogram and concatenates them to form a 3-channel tensor.

    This adapts a single-channel spectrogram (Freq x Time) into a 3-channel input
    (Original, Delta, Delta-Delta) suitable for models like ResNet.

    Args:
        spec_tensor (torch.Tensor): Input spectrogram of shape (1, Freq, Time) or (Freq, Time).
        win_length (int): The window length used for computing deltas.

    Returns:
        torch.Tensor: A 3-channel tensor of shape (3, Freq, Time) containing
                      [original, delta, delta_delta].
    """
    # Ensure input is (1, Freq, Time)
    if spec_tensor.dim() == 2:
        spec_tensor = spec_tensor.unsqueeze(0)

    # spec_tensor is now (C, F, T) where C=1.
    # torchaudio.functional.compute_deltas computes delta along the last dimension (Time) by default.

    # Compute Delta (1st derivative)
    delta = F.compute_deltas(spec_tensor, win_length=win_length)

    # Compute Delta-Delta (2nd derivative)
    delta_delta = F.compute_deltas(delta, win_length=win_length)

    # Concatenate along the channel dimension
    # Result shape: (3, Freq, Time)
    result = torch.cat([spec_tensor, delta, delta_delta], dim=0)

    return result


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).
    Uses 'macro' average for multi-label classification.

    Args:
        y_true (np.array): Ground truth labels (N_samples, N_classes).
        y_pred (np.array): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    try:
        # Check if y_true has at least one positive and one negative sample per class
        # sklearn's roc_auc_score can fail if a class has only one label present in y_true
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # Fallback for edge cases (e.g., during debugging with small batches where a class might be constant)
        # Calculate per-column and ignore columns with only one class present
        n_classes = y_true.shape[1]
        scores = []
        for i in range(n_classes):
            try:
                # Check if column has both 0s and 1s
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                pass

        if scores:
            return np.mean(scores)
        else:
            # If no classes can be evaluated, return 0.5 (random guessing)
            return 0.5


def calculate_pos_weights(df, label_cols):
    """
    Calculates positive class weights for BCEWithLogitsLoss to handle class imbalance.
    Formula: weight = number_of_negatives / number_of_positives

    Args:
        df (pd.DataFrame): DataFrame containing the training labels.
        label_cols (list): List of column names corresponding to the target labels.

    Returns:
        torch.Tensor: Tensor of weights with shape (Num_Classes,).
    """
    weights = []
    total_count = len(df)

    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = total_count - pos_count

        # Avoid division by zero
        if pos_count == 0:
            weight = 1.0
        else:
            weight = neg_count / pos_count

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
