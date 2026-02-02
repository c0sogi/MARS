import os
import numpy as np
import joblib
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import library.config as config
import library.utils as utils


def train_logistic_regression(
    X_train, y_train, X_val, y_val, view_name, save_model=True
):
    """
    Trains a Logistic Regression classifier on the provided embeddings.

    Args:
        X_train (np.ndarray): Training embeddings of shape (N_train, D).
        y_train (np.ndarray): Training labels of shape (N_train,).
        X_val (np.ndarray): Validation embeddings of shape (N_val, D).
        y_val (np.ndarray): Validation labels of shape (N_val,).
        view_name (str): Name of the view (e.g., 'standard', 'global', 'local') for logging and saving.
        save_model (bool): Whether to save the trained model to the working directory.

    Returns:
        tuple: (model, val_probs, val_loss)
            model: The trained LogisticRegression object.
            val_probs: Predicted probabilities on the validation set.
            val_loss: The Log Loss on the validation set.
    """
    print(f"Training Logistic Regression for view: {view_name}...")

    # Ensure labels are integers
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Initialize model with config parameters
    # Solver 'lbfgs' is efficient for multiclass problems with many features
    clf = LogisticRegression(**config.LOGREG_PARAMS)

    # Fit model
    clf.fit(X_train, y_train)

    # Predict on validation set
    val_probs = clf.predict_proba(X_val)

    # Compute metric
    # We pass clf.classes_ to ensure correct column mapping if classes are missing in val
    val_loss = utils.compute_metric(y_val, val_probs, labels=clf.classes_)

    print(f"  View: {view_name} | Validation Log Loss: {val_loss}")

    # Save model
    if save_model:
        model_path = os.path.join(config.WORKING_DIR, f"logreg_{view_name}.joblib")
        joblib.dump(clf, model_path)
        print(f"  Model saved to {model_path}")

    return clf, val_probs, val_loss


def optimize_ensemble_weights(predictions_list, y_true):
    """
    Finds the optimal weights for averaging predictions from multiple models
    to minimize Log Loss on the validation set.

    Args:
        predictions_list (list of np.ndarray): List of probability arrays (N, C).
        y_true (np.ndarray): Ground truth labels (N,).

    Returns:
        np.ndarray: Optimal weights summing to 1.
    """
    print("Optimizing ensemble weights...")

    num_models = len(predictions_list)
    y_true = y_true.astype(int)

    # Objective function to minimize
    def loss_func(weights):
        # Normalize weights to ensure they sum to 1 during optimization steps
        w = weights / np.sum(weights)

        # Weighted average
        final_pred = np.zeros_like(predictions_list[0])
        for i, p in enumerate(predictions_list):
            final_pred += w[i] * p

        # Clip to avoid numerical instability (log(0))
        final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)

        # Renormalize rows to ensure valid probability distribution
        final_pred = final_pred / final_pred.sum(axis=1, keepdims=True)

        return utils.compute_metric(y_true, final_pred)

    # Initial guess: equal weights
    init_weights = np.ones(num_models) / num_models

    # Constraints: sum(weights) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: 0 <= w <= 1 for each weight
    bounds = [(0, 1) for _ in range(num_models)]

    # Optimization using SLSQP
    result = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False},
    )

    opt_weights = result.x

    # Ensure they sum to 1 exactly
    opt_weights = opt_weights / np.sum(opt_weights)

    print(f"  Optimal Weights: {opt_weights}")
    print(f"  Optimized Loss: {result.fun}")

    return opt_weights


def weighted_average_prediction(predictions_list, weights):
    """
    Computes the weighted average of predictions.

    Args:
        predictions_list (list of np.ndarray): List of probability arrays.
        weights (np.ndarray or list): Weights for each prediction array.

    Returns:
        np.ndarray: Weighted average probabilities.
    """
    if len(predictions_list) != len(weights):
        raise ValueError("Number of prediction arrays and weights must match.")

    weighted_pred = np.zeros_like(predictions_list[0])

    for p, w in zip(predictions_list, weights):
        weighted_pred += w * p

    # Renormalize to ensure valid probability distribution (sum to 1 per row)
    # This handles minor floating point errors from the weighted sum
    weighted_pred = weighted_pred / weighted_pred.sum(axis=1, keepdims=True)

    return weighted_pred
