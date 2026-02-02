import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
import library.config as config


def train_logistic_regression(
    X: np.ndarray, y: np.ndarray, stream_name: str, load_cached_model: bool = True
):
    """
    Trains a LogisticRegressionCV model on the provided features and labels.
    Implements caching to save/load the trained model.

    Args:
        X (np.ndarray): Feature matrix of shape (N_samples, N_features).
        y (np.ndarray): Label vector of shape (N_samples,).
        stream_name (str): Identifier for the stream (e.g., 'stream_a', 'stream_b') used for naming the cache file.
        load_cached_model (bool): If True, attempts to load a pre-trained model from disk.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: The trained model.
    """
    # Construct cache path
    model_filename = f"{stream_name}_logreg.joblib"
    model_path = os.path.join(config.WORKING_DIR, model_filename)

    # Attempt to load from cache
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached model from {model_path}...")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"Failed to load cached model: {e}. Retraining...")

    print(f"Training Logistic Regression for {stream_name}...")
    print(f"Input shape: {X.shape}, Labels shape: {y.shape}")

    # Initialize LogisticRegressionCV
    # We use 'neg_log_loss' scoring to select the best C that minimizes log loss.
    clf = LogisticRegressionCV(
        Cs=config.LOGREG_C_VALUES,
        cv=config.LOGREG_CV_FOLDS,
        max_iter=config.LOGREG_MAX_ITER,
        n_jobs=config.LOGREG_JOBS,
        random_state=config.SEED,
        scoring="neg_log_loss",
        multi_class="multinomial",
        verbose=0,  # Keep silent as per requirements
    )

    # Fit the model
    clf.fit(X, y)

    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)

    return clf


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray):
    """
    Evaluates the model on the validation set using Log Loss.

    Args:
        model: Trained sklearn model.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.

    Returns:
        float: The calculated log loss.
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val)

    # Calculate Log Loss
    # labels are indices 0..119, predict_proba returns matrix (N, 120)
    loss = log_loss(y_val, y_pred_proba)

    print(f"Validation Log Loss: {loss}")
    return loss


def predict_probabilities(model, X: np.ndarray):
    """
    Generates probability predictions for the given features.

    Args:
        model: Trained sklearn model.
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Probability matrix of shape (N_samples, N_classes).
    """
    return model.predict_proba(X)
