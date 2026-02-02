import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data_factory import load_train_data, load_test_data
from library.feature_engineering import get_classical_features
from library.training_engine import train_classical_fold, train_neural_fold
from library.stacking import StackingEnsemble


def main():
    # 1. Setup & Config Override
    # Reduce epochs to ensure execution within 2 hours
    Config.EPOCHS = 2
    seed_everything(Config.SEED)

    # 2. Load Data
    train_df = load_train_data(load_cached_data=True)
    test_df = load_test_data()

    # 3. Feature Engineering
    # Returns sparse matrices for TF-IDF and dense arrays for SVD
    train_tfidf, test_tfidf, train_svd, test_svd, train_y = get_classical_features(
        train_df, test_df
    )

    # 4. Cross-Validation Initialization
    # OOF Prediction Storage: (N_samples, 3)
    oof_preds = {
        "lr": np.zeros((len(train_df), 3)),
        "nb": np.zeros((len(train_df), 3)),
        "xgb": np.zeros((len(train_df), 3)),
        "neural": np.zeros((len(train_df), 3)),
    }

    # Test Prediction Storage: List of arrays to be averaged
    test_preds_accum = {k: [] for k in oof_preds}

    # 5. Cross-Validation Loop
    for fold in range(Config.N_FOLDS):
        # Determine indices for this fold
        val_mask = train_df["fold"] == fold
        train_mask = ~val_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # --- Classical Models Training ---
        # Slice features
        X_tfidf_tr = train_tfidf[train_idx]
        X_tfidf_val = train_tfidf[val_idx]

        X_svd_tr = train_svd[train_idx]
        X_svd_val = train_svd[val_idx]

        y_tr = train_y[train_idx]
        y_val = train_y[val_idx]

        # Train and predict
        cls_results = train_classical_fold(
            fold,
            X_tfidf_tr,
            y_tr,
            X_tfidf_val,
            y_val,
            test_tfidf,
            X_svd_tr,
            X_svd_val,
            test_svd,
        )

        # Store results
        for model_name, res in cls_results.items():
            oof_preds[model_name][val_idx] = res["val"]
            test_preds_accum[model_name].append(res["test"])

        # --- Neural Model Training ---
        # Prepare DataFrames for the fold
        tr_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Train and predict
        neural_oof, neural_test = train_neural_fold(
            fold, tr_fold_df, val_fold_df, test_df
        )

        # Store results
        oof_preds["neural"][val_idx] = neural_oof
        test_preds_accum["neural"].append(neural_test)

    # 6. Aggregation of Test Predictions
    test_preds_avg = {k: np.mean(v, axis=0) for k, v in test_preds_accum.items()}

    # 7. Stacking Ensemble
    stacker = StackingEnsemble()
    test_ids = test_df["id"].values

    # Fit meta-learner and generate submission (saved to disk automatically)
    # We ignore the return value here as we need to validate first
    _ = stacker.fit_predict(oof_preds, test_preds_avg, train_y, test_ids)

    # 8. Validation Assessment
    # Re-evaluate on the full OOF set to get the final metric
    X_meta_train = stacker._prepare_meta_features(oof_preds, is_train=True)
    meta_oof_preds = stacker.meta_model.predict_proba(X_meta_train)

    final_metric = calculate_metric(train_y, meta_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 9. Failure Analysis
    # Calculate Cross Entropy per sample
    eps = 1e-15
    clipped_preds = np.clip(meta_oof_preds, eps, 1 - eps)
    rows = np.arange(len(train_y))
    true_class_probs = clipped_preds[rows, train_y]
    sample_losses = -np.log(true_class_probs)

    # Calculate Word Count
    word_counts = train_df["text"].astype(str).apply(lambda x: len(x.split())).values

    # Calculate Correlation
    correlation = np.corrcoef(sample_losses, word_counts)[0, 1]
    print(f"Correlation between Error Magnitude and Word Count: {correlation}")

    # 10. Submission Threshold Check
    THRESHOLD = 0.23237805822413304
    if final_metric >= THRESHOLD:
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
