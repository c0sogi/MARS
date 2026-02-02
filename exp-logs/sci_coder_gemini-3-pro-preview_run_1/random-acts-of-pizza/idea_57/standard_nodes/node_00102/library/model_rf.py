import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config, utils


def train_rf_model(X_train, y_train, X_val, y_val, params=None, save_path=None):
    """
    Trains the Random Forest model using the Interaction-Projected Top-K features.

    Args:
        X_train (scipy.sparse.csr_matrix): Training feature matrix.
        y_train (np.array): Training target vector.
        X_val (scipy.sparse.csr_matrix): Validation feature matrix.
        y_val (np.array): Validation target vector.
        params (dict, optional): Hyperparameters for RandomForestClassifier.
                                 Defaults to config.RF_PARAMS.
        save_path (str, optional): Path to save the trained model pickle.

    Returns:
        tuple: (trained_model, validation_auc_score)
    """
    if params is None:
        params = config.RF_PARAMS

    print("Initializing Random Forest Classifier...")
    # Ensure random state is set from config if not in params
    if "random_state" not in params:
        params["random_state"] = config.RANDOM_STATE

    # Initialize model
    rf_model = RandomForestClassifier(**params)

    print(
        f"Training Random Forest with {X_train.shape[0]} samples and {X_train.shape[1]} features..."
    )
    rf_model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    # Predict probabilities for the positive class (index 1)
    val_probs = rf_model.predict_proba(X_val)[:, 1]

    # Calculate ROC AUC
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Validation ROC AUC: {val_auc}")

    # Save model if path is provided
    if save_path:
        utils.save_pickle(rf_model, save_path)

    return rf_model, val_auc


def predict_rf_model(model, X_test):
    """
    Generates predictions using the trained Random Forest model.

    Args:
        model (RandomForestClassifier): The trained model.
        X_test (scipy.sparse.csr_matrix): Test feature matrix.

    Returns:
        np.array: Predicted probabilities for the positive class.
    """
    print(f"Generating predictions for {X_test.shape[0]} test samples...")
    probs = model.predict_proba(X_test)[:, 1]
    return probs
