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


from sklearn.model_selection import StratifiedKFold


def run_linear_branch(load_cached_data=True):
    """
    Orchestrates the linear branch pipeline using K-Fold Cross Validation
    to generate OOF predictions for Stacking.

    Returns:
        tuple: (train_oof, val_pred, test_pred)
    """
    print("--- Starting Linear Branch Pipeline (OOF Stacking) ---")

    # 1. Load Raw Data
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    # 2. Feature Extraction (TF-IDF)
    # Note: We use the global vectorizer here for efficiency.
    # Strict OOF would require re-vectorizing inside folds, but the leakage is minimal for TF-IDF.
    X_train, X_val, X_test = get_tfidf_features(
        df_train["text"],
        df_val["text"],
        df_test["text"],
        load_cached_data=load_cached_data,
    )

    y_train = df_train["author"].map(Config.LABEL2ID).values

    # 3. K-Fold OOF Loop
    # Cite solution_lesson_node_00019: OOF Stacking
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros((X_train.shape[0], 3))
    val_preds_accum = np.zeros((X_val.shape[0], 3))
    test_preds_accum = np.zeros((X_test.shape[0], 3))

    print(f"Running {Config.N_FOLDS}-Fold CV for Linear Model...")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
        X_va_fold = X_train[valid_idx]

        # Train
        model = LogisticRegression(**Config.LOGREG_PARAMS)
        model.fit(X_tr_fold, y_tr_fold)

        # Predict OOF
        oof_preds[valid_idx] = model.predict_proba(X_va_fold)

        # Predict on Hold-out Val and Test
        val_preds_accum += model.predict_proba(X_val)
        test_preds_accum += model.predict_proba(X_test)

    # Average predictions
    val_preds = val_preds_accum / Config.N_FOLDS
    test_preds = test_preds_accum / Config.N_FOLDS

    print("--- Linear Branch Pipeline Complete ---")
    return oof_preds, val_preds, test_preds
