import numpy as np
from sklearn.metrics import matthews_corrcoef
from library.config import set_seed, FocalLoss


def compute_mcc(y_true, y_prob, threshold=0.5):
    """
    Calculates the Matthews Correlation Coefficient given true labels and predicted probabilities.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_prob (np.ndarray): Predicted probabilities (between 0 and 1).
        threshold (float): Threshold to convert probabilities to binary predictions. Defaults to 0.5.

    Returns:
        float: The Matthews Correlation Coefficient.
    """
    # Convert inputs to numpy arrays to ensure compatibility
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Apply threshold to get binary predictions
    y_pred = (y_prob > threshold).astype(int)

    # Calculate and return MCC
    return matthews_corrcoef(y_true, y_pred)
