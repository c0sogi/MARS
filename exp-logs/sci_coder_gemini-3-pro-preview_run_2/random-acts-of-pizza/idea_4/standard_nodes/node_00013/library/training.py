import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.utils import setup_logger, Timer
from library.feature_extractor import prepare_features
from library.model_definitions import build_linear_branch, build_kernel_branch


def train_and_evaluate(load_cached_data=True, debug=False, logger=None):
    """
    Orchestrates the training, tuning, evaluation, and submission process.

    Args:
        load_cached_data (bool): Whether to load features from cache.
        debug (bool): If True, uses a smaller subset of data and fewer CV splits.
        logger (logging.Logger): Logger instance.
    """
    if logger is None:
        logger = setup_logger("training")

    set_seed(Config.SEED)

    # 1. Load and Prepare Data
    with Timer("Data Loading & Feature Preparation", logger):
        df_train, df_val, df_test = prepare_features(
            load_cached_data=load_cached_data, logger=logger, debug=debug
        )

    # Helper to separate features and target
    def prepare_xy(df, is_test=False):
        drop_cols = ["request_id", "requester_received_pizza"]
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        X = df.drop(columns=cols_to_drop)
        y = (
            df["requester_received_pizza"].values.astype(int)
            if not is_test and "requester_received_pizza" in df.columns
            else None
        )
        return X, y

    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, _ = prepare_xy(df_test, is_test=True)

    # Adjust CV splits for debug mode or extremely small classes
    n_splits = Config.N_SPLITS
    if debug:
        n_splits = 2

    # Safety check for class counts
    min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
    if min_class_count < n_splits:
        n_splits = max(2, min_class_count)
        logger.warning(f"Reduced CV splits to {n_splits} due to class imbalance.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)

    # ==========================================
    # Branch A: Linear Anchor (Logistic Regression)
    # ==========================================
    logger.info("Tuning Branch A: Linear Anchor (Logistic Regression)...")

    best_score_linear = -1.0
    best_params_linear = {}

    for C in Config.LOGREG_GRID["C"]:
        scores = []
        # Manual CV loop to ensure DataFrame integrity for consistency
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr_fold = X_train.iloc[train_idx]
            y_tr_fold = y_train[train_idx]
            X_v_fold = X_train.iloc[val_idx]
            y_v_fold = y_train[val_idx]

            model = build_linear_branch(C=C, random_state=Config.SEED)
            model.fit(X_tr_fold, y_tr_fold)

            preds = model.predict_proba(X_v_fold)[:, 1]
            try:
                score = roc_auc_score(y_v_fold, preds)
                scores.append(score)
            except ValueError:
                scores.append(0.5)

        avg_score = np.mean(scores)
        # logger.info(f"  Linear C={C}: AUC={avg_score:.6f}")

        if avg_score > best_score_linear:
            best_score_linear = avg_score
            best_params_linear = {"C": C}

    logger.info(
        f"Best Linear Params: {best_params_linear} (CV AUC: {best_score_linear:.6f})"
    )

    # ==========================================
    # Branch B: Non-Linear Expert (PLS + SVM)
    # ==========================================
    logger.info("Tuning Branch B: Non-Linear Expert (PLS + SVM)...")

    best_score_svm = -1.0
    best_params_svm = {}

    # Generate grid combinations
    svm_keys = Config.SVM_GRID.keys()
    svm_values = (Config.SVM_GRID[key] for key in svm_keys)
    svm_combinations = [dict(zip(svm_keys, v)) for v in itertools.product(*svm_values)]
    pls_components = Config.PLS_GRID["n_components"]

    for n_comp in pls_components:
        for svm_params in svm_combinations:
            C_svm = svm_params["C"]
            gamma_svm = svm_params["gamma"]

            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr_fold = X_train.iloc[train_idx]
                y_tr_fold = y_train[train_idx]
                X_v_fold = X_train.iloc[val_idx]
                y_v_fold = y_train[val_idx]

                # Build pipeline
                model = build_kernel_branch(
                    pls_n_components=n_comp,
                    svm_C=C_svm,
                    svm_gamma=gamma_svm,
                    random_state=Config.SEED,
                )

                # Fit (Projector fits on X_tr_fold, y_tr_fold; SVC fits/calibrates)
                model.fit(X_tr_fold, y_tr_fold)

                preds = model.predict_proba(X_v_fold)[:, 1]
                try:
                    score = roc_auc_score(y_v_fold, preds)
                    scores.append(score)
                except ValueError:
                    scores.append(0.5)

            avg_score = np.mean(scores)

            if avg_score > best_score_svm:
                best_score_svm = avg_score
                best_params_svm = {
                    "pls_n_components": n_comp,
                    "svm_C": C_svm,
                    "svm_gamma": gamma_svm,
                }

    logger.info(f"Best SVM Params: {best_params_svm} (CV AUC: {best_score_svm:.6f})")

    # ==========================================
    # Validation on Hold-out Set
    # ==========================================
    logger.info("Evaluating on Validation Set (Hold-out)...")

    # Retrain best models on full training set (X_train)
    final_linear = build_linear_branch(
        C=best_params_linear["C"], random_state=Config.SEED
    )
    final_linear.fit(X_train, y_train)

    final_svm = build_kernel_branch(
        pls_n_components=best_params_svm["pls_n_components"],
        svm_C=best_params_svm["svm_C"],
        svm_gamma=best_params_svm["svm_gamma"],
        random_state=Config.SEED,
    )
    final_svm.fit(X_train, y_train)

    # Predict on Validation
    val_pred_linear = final_linear.predict_proba(X_val)[:, 1]
    val_pred_svm = final_svm.predict_proba(X_val)[:, 1]

    # Weighted Ensemble
    w_lin = Config.WEIGHT_LINEAR
    w_svm = Config.WEIGHT_SVM
    val_pred_ensemble = (w_lin * val_pred_linear) + (w_svm * val_pred_svm)

    val_auc_linear = roc_auc_score(y_val, val_pred_linear)
    val_auc_svm = roc_auc_score(y_val, val_pred_svm)
    val_auc_ensemble = roc_auc_score(y_val, val_pred_ensemble)

    logger.info("Validation Metrics:")
    logger.info(f"  Linear AUC:   {val_auc_linear}")
    logger.info(f"  SVM AUC:      {val_auc_svm}")
    logger.info(f"  Ensemble AUC: {val_auc_ensemble}")

    # ==========================================
    # Final Training & Submission
    # ==========================================
    logger.info("Retraining on Full Data (Train + Val) for Submission...")

    # Combine Train and Validation sets
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    # Fit Final Models on Full Data
    final_linear.fit(X_full, y_full)
    final_svm.fit(X_full, y_full)

    # Predict on Test Set
    test_pred_linear = final_linear.predict_proba(X_test)[:, 1]
    test_pred_svm = final_svm.predict_proba(X_test)[:, 1]

    # Ensemble Predictions
    test_pred_ensemble = (w_lin * test_pred_linear) + (w_svm * test_pred_svm)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "request_id": df_test["request_id"],
            "requester_received_pizza": test_pred_ensemble,
        }
    )

    # Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
