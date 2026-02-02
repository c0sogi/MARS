import numpy as np
from sklearn.metrics import log_loss


def compute_log_loss(y_true, y_pred, labels=None):
    """
    Computes the multi-class log loss metric according to task specifications.

    Args:
        y_true (np.ndarray): True class labels (integers) or binary class matrix.
        y_pred (np.ndarray): Predicted probabilities.
        labels (list, optional): List of class labels to index the columns of y_pred.

    Returns:
        float: The computed log loss.
    """
    # 1. Rescale probabilities to sum to 1 (as per task metric description)
    # The submitted probabilities are rescaled prior to being scored.
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities to avoid extremes of log function
    # Task specifies: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Compute Log Loss using sklearn
    loss = log_loss(y_true, y_pred_clipped, labels=labels)

    # 4. Print full precision without rounding
    print(f"Validation Multi-class Log Loss: {loss}")

    return loss


def evaluate_model(model, X_val, y_val, max_samples=None):
    """
    Generates predictions using the model and computes log loss.

    Args:
        model: Trained model object with a predict() method returning probabilities.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        max_samples (int, optional): Limit the number of samples for debugging.

    Returns:
        float: The log loss score.
    """
    # Hyperparameter to control dataset size for debugging
    if max_samples is not None and max_samples > 0:
        if max_samples < len(X_val):
            # print(f"Evaluating on subset of {max_samples} samples...")
            X_val = X_val[:max_samples]
            y_val = y_val[:max_samples]

    # Generate predictions
    # The provided LogisticBaseline.predict returns probabilities (predict_proba)
    y_pred = model.predict(X_val)

    # Retrieve classes from model if available to ensure correct alignment with y_true
    # The LogisticBaseline wrapper stores classes in self.classes_
    labels = getattr(model, "classes_", None)

    return compute_log_loss(y_val, y_pred, labels=labels)
