import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from library.config import Config
from library.utils import set_seed, compute_auc


def train_rf(X_train, y_train, X_val=None, y_val=None, save_path=None):
    """
    Trains the Random Forest model based on Config parameters.

    Args:
        X_train: Training features (sparse or dense).
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        save_path: Path to save the trained model. If None, uses default in WORKING_DIR.

    Returns:
        model: Trained RandomForestClassifier.
        metrics: Dictionary containing validation metrics (if X_val provided).
    """
    set_seed(Config.SEED)

    print("Initializing Random Forest...")
    rf_params = Config.RF_HYPERPARAMETERS.copy()
    model = RandomForestClassifier(**rf_params)

    print(
        f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    model.fit(X_train, y_train)

    metrics = {}
    if X_val is not None and y_val is not None:
        print("Evaluating on validation set...")
        # Predict probabilities for the positive class
        val_probs = model.predict_proba(X_val)[:, 1]
        auc_score = compute_auc(y_val, val_probs)
        metrics["auc"] = auc_score
        print(f"Validation AUC: {auc_score}")

    # Save model
    if save_path is None:
        save_path = os.path.join(Config.WORKING_DIR, "rf_model.joblib")

    print(f"Saving RF model to {save_path}...")
    joblib.dump(model, save_path)

    return model, metrics


def predict_rf(model, X_test):
    """
    Generates predictions using the trained Random Forest model.

    Args:
        model: Trained RandomForestClassifier.
        X_test: Test features.

    Returns:
        probs: Probability estimates for the positive class.
    """
    # Predict probabilities for class 1
    probs = model.predict_proba(X_test)[:, 1]
    return probs


def run_rf_pipeline(processed_data):
    """
    Orchestrates the RF pipeline using the dictionary output from FeatureProcessor.

    Args:
        processed_data: Dictionary containing split features (e.g., 'train_rf', 'train_y').

    Returns:
        model: Trained model.
        val_probs: Validation predictions.
        test_probs: Test predictions.
    """
    # Extract Data
    X_train = processed_data["train_rf"]
    y_train = processed_data["train_y"]
    X_val = processed_data["val_rf"]
    y_val = processed_data["val_y"] if "val_y" in processed_data else None
    X_test = processed_data["test_rf"]

    # Train
    model, metrics = train_rf(X_train, y_train, X_val, y_val)

    # Inference
    val_probs = None
    if X_val is not None:
        val_probs = predict_rf(model, X_val)

    test_probs = predict_rf(model, X_test)

    return model, val_probs, test_probs
