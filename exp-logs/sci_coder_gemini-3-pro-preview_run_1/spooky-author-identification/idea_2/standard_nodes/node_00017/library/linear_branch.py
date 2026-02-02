import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import compute_log_loss
from library.data_loader import load_data, get_tfidf_features


def run_linear_branch(load_cached_data=True):
    """
    Orchestrates the linear branch pipeline with K-Fold Cross Validation.
    Returns OOF predictions for training data, and averaged predictions for Val/Test.
    """
    print("--- Starting Linear Branch Pipeline (K-Fold) ---")

    # 1. Load Raw Data
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    # 2. Feature Extraction
    X_train, X_val, X_test = get_tfidf_features(
        df_train["text"],
        df_val["text"],
        df_test["text"],
        load_cached_data=load_cached_data,
    )

    y_train = df_train["author"].map(Config.LABEL2ID).values
    y_val = df_val["author"].map(Config.LABEL2ID).values

    # 3. K-Fold CV
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros((len(df_train), 3))
    val_preds_accum = np.zeros((len(df_val), 3))
    test_preds_accum = np.zeros((len(df_test), 3))

    print(f"Training Linear Model with {Config.N_FOLDS} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va = X_train[val_idx]

        model = LogisticRegression(**Config.LOGREG_PARAMS)
        model.fit(X_tr, y_tr)

        # OOF Prediction
        oof_preds[val_idx] = model.predict_proba(X_va)

        # Accumulate Val/Test Predictions
        val_preds_accum += model.predict_proba(X_val)
        test_preds_accum += model.predict_proba(X_test)

    # Average predictions
    val_probs = val_preds_accum / Config.N_FOLDS
    test_probs = test_preds_accum / Config.N_FOLDS

    loss = compute_log_loss(y_train, oof_preds, labels=[0, 1, 2])
    print(f"Linear Branch OOF Log Loss: {loss}")

    print("--- Linear Branch Pipeline Complete ---")
    return oof_preds, val_probs, test_probs, y_train
