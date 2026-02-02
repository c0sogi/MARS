import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from library.config import WORKING_DIR, SEED, N_FOLDS, NUM_WORKERS
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import load_data, get_tfidf_features, get_svd_features


def run_classical_cv(load_cached_preds=True):
    """
    Orchestrates the training and prediction of classical models (LR, NB, XGB)
    using Stratified K-Fold Cross-Validation.

    Args:
        load_cached_preds (bool): If True, attempts to load predictions from disk.

    Returns:
        dict: A dictionary containing OOF and Test predictions for each model.
              Keys: 'lr_oof', 'lr_test', 'nb_oof', 'nb_test', 'xgb_oof', 'xgb_test'.
    """
    seed_everything(SEED)

    # Define paths for caching predictions
    cache_files = {
        "lr_oof": os.path.join(WORKING_DIR, "oof_lr.npy"),
        "lr_test": os.path.join(WORKING_DIR, "pred_test_lr.npy"),
        "nb_oof": os.path.join(WORKING_DIR, "oof_nb.npy"),
        "nb_test": os.path.join(WORKING_DIR, "pred_test_nb.npy"),
        "xgb_oof": os.path.join(WORKING_DIR, "oof_xgb.npy"),
        "xgb_test": os.path.join(WORKING_DIR, "pred_test_xgb.npy"),
    }

    # Check if all cache files exist
    if load_cached_preds and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached classical model predictions...")
        results = {}
        for k, v in cache_files.items():
            results[k] = np.load(v)
        return results

    print("Running Classical Models CV...")

    # 1. Load Data
    train_df, val_df, test_df = load_data()

    # 2. Prepare Labels
    # We combine train and val to perform 5-fold CV on the maximum available labeled data
    le = LabelEncoder()
    le.fit(train_df["author"])  # Fits EAP, HPL, MWS -> 0, 1, 2

    y_train = le.transform(train_df["author"])
    y_val = le.transform(val_df["author"])
    y_full = np.concatenate([y_train, y_val])

    # 3. Prepare Features
    # Load features (internally cached by data_loader)
    train_tfidf, val_tfidf, test_tfidf = get_tfidf_features(
        train_df["text"], val_df["text"], test_df["text"]
    )
    train_svd, val_svd, test_svd = get_svd_features(train_tfidf, val_tfidf, test_tfidf)

    # Stack features to match y_full
    X_tfidf_full = sp.vstack([train_tfidf, val_tfidf], format="csr")
    X_svd_full = np.vstack([train_svd, val_svd])

    # 4. Initialize Prediction Containers
    n_samples = len(y_full)
    n_test = len(test_df)
    n_classes = 3

    # Dictionaries to hold results
    results = {
        "lr_oof": np.zeros((n_samples, n_classes)),
        "lr_test": np.zeros((n_test, n_classes)),
        "nb_oof": np.zeros((n_samples, n_classes)),
        "nb_test": np.zeros((n_test, n_classes)),
        "xgb_oof": np.zeros((n_samples, n_classes)),
        "xgb_test": np.zeros((n_test, n_classes)),
    }

    # 5. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"Starting {N_FOLDS}-Fold Stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y_full)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Split Labels
        y_tr, y_va = y_full[train_idx], y_full[val_idx]

        # ---------------------------
        # Model 1: Logistic Regression (TF-IDF)
        # ---------------------------
        X_tr_tfidf, X_va_tfidf = X_tfidf_full[train_idx], X_tfidf_full[val_idx]

        clf_lr = LogisticRegression(
            C=1.0,
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=SEED,
            max_iter=1000,
        )
        clf_lr.fit(X_tr_tfidf, y_tr)

        p_val_lr = clf_lr.predict_proba(X_va_tfidf)
        p_test_lr = clf_lr.predict_proba(test_tfidf)

        results["lr_oof"][val_idx] = p_val_lr
        results["lr_test"] += p_test_lr / N_FOLDS

        print(f"LR Log Loss: {calculate_log_loss(y_va, p_val_lr)}")

        # ---------------------------
        # Model 2: Multinomial Naive Bayes (TF-IDF)
        # ---------------------------
        clf_nb = MultinomialNB(alpha=0.01)
        clf_nb.fit(X_tr_tfidf, y_tr)

        p_val_nb = clf_nb.predict_proba(X_va_tfidf)
        p_test_nb = clf_nb.predict_proba(test_tfidf)

        results["nb_oof"][val_idx] = p_val_nb
        results["nb_test"] += p_test_nb / N_FOLDS

        print(f"NB Log Loss: {calculate_log_loss(y_va, p_val_nb)}")

        # ---------------------------
        # Model 3: XGBoost (SVD)
        # ---------------------------
        X_tr_svd, X_va_svd = X_svd_full[train_idx], X_svd_full[val_idx]

        clf_xgb = XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            n_jobs=-1,
            random_state=SEED,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
        )

        clf_xgb.fit(X_tr_svd, y_tr, eval_set=[(X_va_svd, y_va)], verbose=False)

        p_val_xgb = clf_xgb.predict_proba(X_va_svd)
        p_test_xgb = clf_xgb.predict_proba(test_svd)

        results["xgb_oof"][val_idx] = p_val_xgb
        results["xgb_test"] += p_test_xgb / N_FOLDS

        print(f"XGB Log Loss: {calculate_log_loss(y_va, p_val_xgb)}")

    # 6. Overall Evaluation
    print("\n--- Overall CV Scores ---")
    print(f"LR OOF Log Loss: {calculate_log_loss(y_full, results['lr_oof'])}")
    print(f"NB OOF Log Loss: {calculate_log_loss(y_full, results['nb_oof'])}")
    print(f"XGB OOF Log Loss: {calculate_log_loss(y_full, results['xgb_oof'])}")

    # 7. Save Predictions to Cache
    print("Saving predictions to cache...")
    np.save(cache_files["lr_oof"], results["lr_oof"])
    np.save(cache_files["lr_test"], results["lr_test"])
    np.save(cache_files["nb_oof"], results["nb_oof"])
    np.save(cache_files["nb_test"], results["nb_test"])
    np.save(cache_files["xgb_oof"], results["xgb_oof"])
    np.save(cache_files["xgb_test"], results["xgb_test"])

    return results
