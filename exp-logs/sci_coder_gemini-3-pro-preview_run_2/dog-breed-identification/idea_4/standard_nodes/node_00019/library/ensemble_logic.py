import os
import numpy as np
import joblib
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import library.config as config
import library.utils as utils


def train_logreg_cv(
    X_train, y_train, X_val, y_val, name="early_fusion", save_model=True
):
    """
    Trains a Logistic Regression classifier with automatic hyperparameter tuning (CV).

    Args:
        X_train (np.ndarray): Training embeddings.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation embeddings.
        y_val (np.ndarray): Validation labels.
        name (str): Name for saving the model.
        save_model (bool): Whether to save the model.

    Returns:
        tuple: (model, val_probs, val_loss)
    """
    print(f"Training LogisticRegressionCV for {name}...")

    # Ensure labels are integers
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Use LogisticRegressionCV to automatically tune C
    # Cs=10 tries 10 values on log scale (1e-4 to 1e4)
    # cv=5 uses 5-fold stratified CV
    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="neg_log_loss",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        random_state=config.SEED,
        multi_class="multinomial",
    )

    # Fit model
    clf.fit(X_train, y_train)

    print(
        f"  Best C: {clf.C_[0]}"
    )  # Print best C for the first class (usually same for all in multinomial)

    # Predict on validation set
    val_probs = clf.predict_proba(X_val)

    # Compute metric
    val_loss = utils.compute_metric(y_val, val_probs, labels=clf.classes_)
    print(f"  {name} | Validation Log Loss: {val_loss}")

    # Save model
    if save_model:
        model_path = os.path.join(config.WORKING_DIR, f"logreg_{name}.joblib")
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
