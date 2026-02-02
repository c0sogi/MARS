import numpy as np
import torch
from library.config import seed_everything

# Alias the imported function to meet the requirement of providing `set_seed`
set_seed = seed_everything


def competition_log_loss(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the weighted multi-label logarithmic loss for the RSNA Cervical Spine Fracture Detection task.

    The metric is defined as the weighted average of the log loss for each label.
    Weights:
        - patient_overall: 1.0
        - C1 to C7: 1/7 each

    Args:
        y_true (np.ndarray): Binary ground truth labels of shape (N_samples, 8).
                             Columns expected order: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        y_pred (np.ndarray): Predicted probabilities of shape (N_samples, 8).
                             Columns expected order: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        epsilon (float): Small value to prevent log(0).

    Returns:
        float: The weighted log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    # Clip predictions to avoid log(0) and log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Define weights
    # C1-C7 get weight 1/7, patient_overall gets weight 1.0
    # We assume the last column is patient_overall based on the 8-column structure
    weights = np.array([1 / 7] * 7 + [1.0])

    # Calculate Binary Cross Entropy for each element
    # L = -[y * log(p) + (1-y) * log(1-p)]
    loss_per_element = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights to each column
    weighted_loss = loss_per_element * weights

    # The competition metric is averaged across "all rows" in the submission file.
    # In our matrix formulation, this is equivalent to the mean of the weighted loss
    # if we consider the weights as scaling factors for the importance of each 'row'.
    # However, strictly speaking, the submission has 8 rows per exam.
    # We calculate the mean over all samples and all columns.

    return np.mean(weighted_loss)
