import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import compute_log_loss
from library.data_loader import load_data, get_tfidf_features


def train_linear_model(
    X_train, y_train, X_val=None, y_val=None, save_path=Config.LINEAR_MODEL_PATH
):
    """
    Trains the Logistic Regression model on sparse features.

    Args:
        X_train (scipy.sparse.csr_matrix): Training features.
        y_train (array-like): Training labels (encoded as integers).
        X_val (scipy.sparse.csr_matrix, optional): Validation features for scoring.
        y_val (array-like, optional): Validation labels for scoring.
        save_path (str): Path to save the trained model.

    Returns:
        model: The trained LogisticRegression model.
    """
    print("Initializing Linear Model (Logistic Regression)...")
    model = LogisticRegression(**Config.LOGREG_PARAMS)

    print(f"Fitting model on training data with shape {X_train.shape}...")
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        print("Evaluating on validation set...")
        val_probs = model.predict_proba(X_val)
        # Ensure classes are mapped correctly for log loss
        # model.classes_ should be [0, 1, 2] corresponding to EAP, HPL, MWS
        loss = compute_log_loss(y_val, val_probs, labels=model.classes_)
        print(f"Validation Log Loss: {loss}")

    if save_path:
        print(f"Saving linear model to {save_path}...")
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    return model


def predict_linear(model, X):
    """
    Generates probability predictions using the linear model.

    Args:
        model: Trained LogisticRegression model.
        X (scipy.sparse.csr_matrix): Features to predict on.

    Returns:
        numpy.ndarray: Predicted probabilities of shape (n_samples, n_classes).
    """
    return model.predict_proba(X)


def run_linear_branch(load_cached_data=True):
    """
    Orchestrates the linear branch pipeline: loading data, feature extraction,
    training (or loading) the model, and generating predictions.

    Args:
        load_cached_data (bool): Whether to use cached features and model.

    Returns:
        tuple: (val_probs, test_probs, y_val)
            - val_probs: Predictions for validation set.
            - test_probs: Predictions for test set.
            - y_val: Ground truth labels for validation set (integers).
    """
    print("--- Starting Linear Branch Pipeline ---")

    # 1. Load Raw Data
    print("Loading raw metadata...")
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    # 2. Feature Extraction (TF-IDF)
    print("Generating/Loading TF-IDF features...")
    # Pass Series objects to the vectorizer helper
    X_train, X_val, X_test = get_tfidf_features(
        df_train["text"],
        df_val["text"],
        df_test["text"],
        load_cached_data=load_cached_data,
    )

    # 3. Prepare Labels
    print("Encoding labels...")
    # Map string labels to IDs using Config
    y_train = df_train["author"].map(Config.LABEL2ID).values
    y_val = df_val["author"].map(Config.LABEL2ID).values

    # 4. Train or Load Model
    model = None
    if load_cached_data and os.path.exists(Config.LINEAR_MODEL_PATH):
        print(f"Loading existing linear model from {Config.LINEAR_MODEL_PATH}...")
        try:
            loaded_model = joblib.load(Config.LINEAR_MODEL_PATH)
            # Check if model feature count matches data feature count
            if (
                hasattr(loaded_model, "n_features_in_")
                and loaded_model.n_features_in_ != X_train.shape[1]
            ):
                print(
                    f"Model expects {loaded_model.n_features_in_} features but data has {X_train.shape[1]}. Retraining..."
                )
                model = None
            else:
                model = loaded_model
        except Exception as e:
            print(f"Failed to load model: {e}. Proceeding to retrain.")

    if model is None:
        model = train_linear_model(X_train, y_train, X_val, y_val)

    # 5. Generate Predictions
    print("Generating predictions...")
    val_probs = predict_linear(model, X_val)
    test_probs = predict_linear(model, X_test)

    print("--- Linear Branch Pipeline Complete ---")
    return val_probs, test_probs, y_val
