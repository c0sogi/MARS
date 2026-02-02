import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, ensure_dir


def train_rf_model(X_train, y_train, X_val=None, y_val=None, save_path=None):
    """
    Trains the Random Forest model (Stream A) using parameters from Config.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray, optional): Validation feature matrix.
        y_val (np.ndarray, optional): Validation labels.
        save_path (str, optional): File path to save the trained model (using joblib).

    Returns:
        model: The trained RandomForestClassifier instance.
    """
    # Set seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    print("Initializing Random Forest Classifier...")
    print(f"  Estimators: {Config.RF_N_ESTIMATORS}")
    print(f"  Class Weight: {Config.RF_CLASS_WEIGHT}")
    print(f"  Min Samples Leaf: {Config.RF_MIN_SAMPLES_LEAF}")

    model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.RANDOM_STATE,
        verbose=0,
    )

    print(
        f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        print("Evaluating Random Forest on validation set...")
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        # Print full precision as requested
        print(f"Validation ROC AUC: {val_auc}")

    if save_path:
        ensure_dir(save_path)
        print(f"Saving Random Forest model to {save_path}...")
        joblib.dump(model, save_path)

    return model


def predict_rf(model, X):
    """
    Generates probability predictions using the trained Random Forest model.

    Args:
        model: Trained RandomForestClassifier.
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Probabilities for the positive class.
    """
    # Ensure input is 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Predict probabilities for the positive class (index 1)
    probs = model.predict_proba(X)[:, 1]
    return probs
