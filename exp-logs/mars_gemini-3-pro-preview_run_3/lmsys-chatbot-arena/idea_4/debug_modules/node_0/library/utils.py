import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from library.config import seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the seed_everything function from the configuration library.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def compute_score(y_true, y_pred):
    """
    Calculates the competition metric (Log Loss) and auxiliary metric (Accuracy).

    Args:
        y_true (array-like): Ground truth targets. Shape (N, 3).
                             Can be one-hot encoded or soft probabilities.
        y_pred (array-like): Predicted probabilities. Shape (N, 3).

    Returns:
        dict: A dictionary containing the calculated metrics:
              - 'log_loss': The log loss value.
              - 'accuracy': The accuracy score based on argmax.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Log Loss
    # scikit-learn's log_loss handles both label encoding and probability targets.
    # It automatically clips predictions to avoid log(0).
    loss = log_loss(y_true, y_pred)

    # Calculate Accuracy
    # We determine the class by taking the index of the maximum probability (argmax).
    # This works for both hard labels and soft probability distributions.
    true_indices = np.argmax(y_true, axis=1)
    pred_indices = np.argmax(y_pred, axis=1)
    acc = accuracy_score(true_indices, pred_indices)

    return {"log_loss": loss, "accuracy": acc}
