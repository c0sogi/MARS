import os
import sys
import itertools
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
from library.config import Config, set_seed
from library.utils import setup_logger
from library.feature_extractor import prepare_features
from library.model_definitions import build_linear_branch, build_kernel_branch


def get_Xy(df, is_test=False):
    """Helper to separate features and target."""
    drop_cols = ["request_id", "requester_received_pizza"]
    # Drop columns that exist
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = None
    if not is_test and "requester_received_pizza" in df.columns:
        y = df["requester_received_pizza"].values.astype(int)
    return X, y


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)
    warnings.filterwarnings("ignore")

    # 2. Data Loading & Feature Preparation
    logger.info("Loading and preparing features...")
    # prepare_features handles caching and GPU usage for embeddings internally
    df_train, df_val, df_test = prepare_features(load_cached_data=True, logger=logger)

    # 3. Prepare Matrices
    X_train, y_train = get_Xy(df_train)
    X_val, y_val = get_Xy(df_val)
    X_test, _ = get_Xy(df_test, is_test=True)

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
    )

    # 4. Hyperparameter Tuning

    # --- Branch A: Linear Anchor ---
    logger.info("Tuning Branch A (Linear)...")
    best_lin_score = -1.0
    best_lin_params = {}

    for C in Config.LOGREG_GRID["C"]:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

            model = build_linear_branch(C=C, random_state=Config.SEED)
            model.fit(X_tr, y_tr)

            # Predict
            preds = model.predict_proba(X_va)[:, 1]
            try:
                scores.append(roc_auc_score(y_va, preds))
            except ValueError:
                scores.append(0.5)

        avg_score = np.mean(scores)
        if avg_score > best_lin_score:
            best_lin_score = avg_score
            best_lin_params = {"C": C}

    logger.info(f"Best Linear Params: {best_lin_params} (CV AUC: {best_lin_score:.6f})")

    # --- Branch B: Non-Linear Expert (PLS + SVM) ---
    logger.info("Tuning Branch B (PLS + SVM)...")
    best_svm_score = -1.0
    best_svm_params = {}

    # Prepare grid combinations
    svm_keys = Config.SVM_GRID.keys()
    svm_values = (Config.SVM_GRID[key] for key in svm_keys)
    svm_combinations = [dict(zip(svm_keys, v)) for v in itertools.product(*svm_values)]

    for n_comp in Config.PLS_GRID["n_components"]:
        for svm_p in svm_combinations:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

                model = build_kernel_branch(
                    pls_n_components=n_comp,
                    svm_C=svm_p["C"],
                    svm_gamma=svm_p["gamma"],
                    random_state=Config.SEED,
                )
                model.fit(X_tr, y_tr)

                preds = model.predict_proba(X_va)[:, 1]
                try:
                    scores.append(roc_auc_score(y_va, preds))
                except ValueError:
                    scores.append(0.5)

            avg_score = np.mean(scores)
            if avg_score > best_svm_score:
                best_svm_score = avg_score
                best_svm_params = {
                    "pls_n_components": n_comp,
                    "svm_C": svm_p["C"],
                    "svm_gamma": svm_p["gamma"],
                }

    logger.info(f"Best SVM Params: {best_svm_params} (CV AUC: {best_svm_score:.6f})")

    # 5. Validation on Hold-out Set
    logger.info("Evaluating on Hold-out Validation Set...")

    # Retrain best models on full training set
    final_linear = build_linear_branch(C=best_lin_params["C"], random_state=Config.SEED)
    final_linear.fit(X_train, y_train)

    final_svm = build_kernel_branch(
        pls_n_components=best_svm_params["pls_n_components"],
        svm_C=best_svm_params["svm_C"],
        svm_gamma=best_svm_params["svm_gamma"],
        random_state=Config.SEED,
    )
    final_svm.fit(X_train, y_train)

    # Predict
    val_pred_linear = final_linear.predict_proba(X_val)[:, 1]
    val_pred_svm = final_svm.predict_proba(X_val)[:, 1]

    # Ensemble
    val_pred_ensemble = (Config.WEIGHT_LINEAR * val_pred_linear) + (
        Config.WEIGHT_SVM * val_pred_svm
    )

    val_auc = roc_auc_score(y_val, val_pred_ensemble)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_pred_ensemble)

    # Correlate error with features (focusing on metadata for interpretability)
    # We exclude embedding columns ("emb_") from the printout to avoid spam
    meta_cols = [c for c in X_val.columns if not str(c).startswith("emb_")]

    correlations = {}
    for col in meta_cols:
        # Ensure column is numeric and has variance
        if pd.api.types.is_numeric_dtype(X_val[col]) and X_val[col].std() > 0:
            corr = np.corrcoef(X_val[col], errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Prediction Error:")
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 7. Submission
    threshold = 0.6994047619047619
    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc:.6f}) > Threshold ({threshold:.6f}). Generating Submission..."
        )

        # Combine Train + Val
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Refit on Full Data
        final_linear.fit(X_full, y_full)
        final_svm.fit(X_full, y_full)

        # Predict on Test
        test_pred_linear = final_linear.predict_proba(X_test)[:, 1]
        test_pred_svm = final_svm.predict_proba(X_test)[:, 1]

        test_pred_ensemble = (Config.WEIGHT_LINEAR * test_pred_linear) + (
            Config.WEIGHT_SVM * test_pred_svm
        )

        # Save
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": test_pred_ensemble,
            }
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation AUC ({val_auc:.6f}) <= Threshold ({threshold:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
