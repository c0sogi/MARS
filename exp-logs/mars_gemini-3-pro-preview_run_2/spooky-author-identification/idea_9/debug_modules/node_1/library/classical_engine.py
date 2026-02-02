import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import torch

from library.config import Config
from library.utils import seed_everything, get_score


def run_classical_cv(data_dict, train_df, val_df, load_cached_preds=True):
    """
    Runs Stratified K-Fold CV for Classical Models (LR, NB, XGB).
    Generates OOF predictions and Test predictions.

    Args:
        data_dict (dict): Dictionary containing sparse and dense features.
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        load_cached_preds (bool): If True, attempts to load predictions from disk.

    Returns:
        tuple: (oof_preds_dict, test_preds_dict)
    """
    seed_everything(Config.seed)

    # Define file paths for caching
    cache_files = {
        "oof_lr": os.path.join(Config.output_dir, "oof_lr.npy"),
        "test_lr": os.path.join(Config.output_dir, "pred_test_lr.npy"),
        "oof_nb": os.path.join(Config.output_dir, "oof_nb.npy"),
        "test_nb": os.path.join(Config.output_dir, "pred_test_nb.npy"),
        "oof_xgb": os.path.join(Config.output_dir, "oof_xgb.npy"),
        "test_xgb": os.path.join(Config.output_dir, "pred_test_xgb.npy"),
    }

    # Check cache
    all_exist = all(os.path.exists(p) for p in cache_files.values())
    if load_cached_preds and all_exist:
        print("Loading classical model predictions from cache...")
        oof_preds = {
            "lr": np.load(cache_files["oof_lr"]),
            "nb": np.load(cache_files["oof_nb"]),
            "xgb": np.load(cache_files["oof_xgb"]),
        }
        test_preds = {
            "lr": np.load(cache_files["test_lr"]),
            "nb": np.load(cache_files["test_nb"]),
            "xgb": np.load(cache_files["test_xgb"]),
        }
        return oof_preds, test_preds

    print("Training classical models from scratch...")

    # 1. Prepare Data (Concatenate Train + Val for Full CV)
    # We combine the provided splits to perform a full K-Fold CV on all labeled data.

    # Labels
    y_train = train_df["author"].map(Config.label2id).values
    y_val = val_df["author"].map(Config.label2id).values
    y_full = np.concatenate([y_train, y_val])

    # Features - Sparse (for LR, NB)
    X_sparse_full = sparse.vstack([data_dict["train_sparse"], data_dict["val_sparse"]])
    X_sparse_test = data_dict["test_sparse"]

    # Features - Dense (for XGB)
    X_dense_full = np.concatenate(
        [data_dict["train_dense"], data_dict["val_dense"]], axis=0
    )
    X_dense_test = data_dict["test_dense"]

    # 2. Initialize Storage
    n_samples = len(y_full)
    n_test = X_sparse_test.shape[0]
    n_classes = Config.num_classes

    oof_preds = {
        "lr": np.zeros((n_samples, n_classes)),
        "nb": np.zeros((n_samples, n_classes)),
        "xgb": np.zeros((n_samples, n_classes)),
    }

    test_preds = {
        "lr": np.zeros((n_test, n_classes)),
        "nb": np.zeros((n_test, n_classes)),
        "xgb": np.zeros((n_test, n_classes)),
    }

    # 3. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Determine device for XGBoost
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(np.zeros(n_samples), y_full)
    ):
        # Split Labels
        y_tr, y_va = y_full[train_idx], y_full[valid_idx]

        # --- Logistic Regression ---
        X_tr_sp, X_va_sp = X_sparse_full[train_idx], X_sparse_full[valid_idx]

        clf_lr = LogisticRegression(
            C=1.0,
            solver="liblinear",
            multi_class="ovr",
            random_state=Config.seed,
            max_iter=1000,
        )
        clf_lr.fit(X_tr_sp, y_tr)

        p_va_lr = clf_lr.predict_proba(X_va_sp)
        oof_preds["lr"][valid_idx] = p_va_lr
        test_preds["lr"] += clf_lr.predict_proba(X_sparse_test) / Config.n_folds

        # --- Naive Bayes ---
        clf_nb = MultinomialNB(alpha=0.02)
        clf_nb.fit(X_tr_sp, y_tr)

        p_va_nb = clf_nb.predict_proba(X_va_sp)
        oof_preds["nb"][valid_idx] = p_va_nb
        test_preds["nb"] += clf_nb.predict_proba(X_sparse_test) / Config.n_folds

        # --- XGBoost ---
        X_tr_dn, X_va_dn = X_dense_full[train_idx], X_dense_full[valid_idx]

        clf_xgb = xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=n_classes,
            random_state=Config.seed,
            n_jobs=-1,
            device=xgb_device,
            early_stopping_rounds=50,
            verbosity=0,
        )

        clf_xgb.fit(
            X_tr_dn,
            y_tr,
            eval_set=[(X_va_dn, y_va)],
            verbose=False,
        )

        p_va_xgb = clf_xgb.predict_proba(X_va_dn)
        oof_preds["xgb"][valid_idx] = p_va_xgb
        test_preds["xgb"] += clf_xgb.predict_proba(X_dense_test) / Config.n_folds

        # Logging
        score_lr = get_score(y_va, p_va_lr)
        score_nb = get_score(y_va, p_va_nb)
        score_xgb = get_score(y_va, p_va_xgb)

        print(f"Fold {fold+1} | LR: {score_lr} | NB: {score_nb} | XGB: {score_xgb}")

    # 4. Save Results
    print("Saving classical predictions...")
    np.save(cache_files["oof_lr"], oof_preds["lr"])
    np.save(cache_files["test_lr"], test_preds["lr"])
    np.save(cache_files["oof_nb"], oof_preds["nb"])
    np.save(cache_files["test_nb"], test_preds["nb"])
    np.save(cache_files["oof_xgb"], oof_preds["xgb"])
    np.save(cache_files["test_xgb"], test_preds["xgb"])

    # Also save the full label set for alignment verification in the meta-learner
    np.save(os.path.join(Config.output_dir, "train_labels_full.npy"), y_full)

    return oof_preds, test_preds
