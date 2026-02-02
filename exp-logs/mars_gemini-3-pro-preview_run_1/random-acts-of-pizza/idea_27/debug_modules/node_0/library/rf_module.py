import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config, utils


def train_rf(X_train, y_train, X_val=None, y_val=None):
    """
    Trains a Random Forest classifier using the configuration specified in config.py.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray, optional): Validation features.
        y_val (np.ndarray, optional): Validation labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    # Ensure reproducibility
    utils.set_seed()

    print("Initializing Random Forest with parameters:")
    print(config.RF_PARAMS)

    # Initialize model with parameters from config (Low-Bias configuration)
    model = RandomForestClassifier(**config.RF_PARAMS)

    print("Fitting Random Forest...")
    model.fit(X_train, y_train)

    # Validation evaluation
    if X_val is not None and y_val is not None:
        print("Evaluating Random Forest on validation set...")
        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        # Print full precision as requested
        print(f"RF Validation AUC: {auc}")

    return model


def predict_rf(model, X):
    """
    Generates probability predictions using the trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        X (np.ndarray): Features to predict on.

    Returns:
        np.ndarray: Predicted probabilities for the positive class.
    """
    # Return probabilities for class 1 (received pizza)
    return model.predict_proba(X)[:, 1]


def run_rf_pipeline(rf_data):
    """
    Orchestrates the Random Forest training and prediction pipeline.

    Args:
        rf_data (dict): Dictionary containing X_train, y_train, X_val, y_val, X_test.

    Returns:
        tuple: (val_preds, test_preds, model)
    """
    print("Starting Random Forest Pipeline...")

    X_train = rf_data["X_train"]
    y_train = rf_data["y_train"]
    X_val = rf_data["X_val"]
    y_val = rf_data["y_val"]
    X_test = rf_data["X_test"]

    # Train the model
    model = train_rf(X_train, y_train, X_val, y_val)

    # Generate predictions
    val_preds = predict_rf(model, X_val)
    test_preds = predict_rf(model, X_test)

    # Save the model artifact
    model_path = os.path.join(config.WORKING_DIR, "rf_model.joblib")
    joblib.dump(model, model_path)
    print(f"Random Forest model saved to {model_path}")

    return val_preds, test_preds, model
