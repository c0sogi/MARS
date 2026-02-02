import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

# Import library modules
from library.config import SEED, WORKING_DIR, SUBMISSION_PATH
import library.config as config_module
from library.utils import seed_everything
from library.data_loader import DataLoader
from library.training_pipeline import TrainingPipeline, NON_FEATURE_COLS
from library.inference_pipeline import InferencePipeline


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    seed_everything(SEED)

    # Configure models for Fast Baseline and GPU acceleration
    # Patching config parameters at runtime
    config_module.LGBM_PARAMS["n_estimators"] = 200
    config_module.XGB_PARAMS["n_estimators"] = 200

    if torch.cuda.is_available():
        config_module.XGB_PARAMS["tree_method"] = "hist"
        config_module.XGB_PARAMS["device"] = "cuda"
        config_module.LGBM_PARAMS["device"] = "gpu"

    # =========================================================================
    # 2. Data Loading & Downsampling
    # =========================================================================
    # Load full datasets (cached if available)
    train_df = DataLoader.load_train_data(load_cached_data=True)
    val_df = DataLoader.load_val_data(load_cached_data=True)

    # Downsample training data for speed
    # Strategy: Keep all positives, sample negatives to reach ~100k total rows
    pos_mask = train_df["contact"] == 1
    neg_mask = train_df["contact"] == 0

    pos_df = train_df[pos_mask]
    neg_df = train_df[neg_mask]

    # Calculate negative sample size
    n_pos = len(pos_df)
    target_total = 100000
    n_neg = max(0, min(len(neg_df), target_total - n_pos))

    neg_df_sampled = neg_df.sample(n=n_neg, random_state=SEED)

    # Combine and shuffle
    train_df_sampled = (
        pd.concat([pos_df, neg_df_sampled])
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )

    # =========================================================================
    # 3. Training Pipeline
    # =========================================================================

    # Phase 1: Train Scouts
    # Uses the sampled training set and full validation set
    scout_models = TrainingPipeline.train_scouts(train_df_sampled, val_df)

    # Phase 2: Mine Hard Negatives
    # We mine from the sampled training set.
    # load_cached_data=False ensures we compute indices for the current sampled dataframe.
    hard_neg_indices = TrainingPipeline.mine_hard_negatives(
        scout_models, train_df_sampled, load_cached_data=False
    )

    # Phase 3: Train Experts
    # Trains on Positives + Hard Negatives + Anchors
    expert_models = TrainingPipeline.train_experts(
        train_df_sampled, hard_neg_indices, val_df
    )

    # =========================================================================
    # 4. Evaluation & Threshold Optimization
    # =========================================================================
    # Identify feature columns
    feature_cols = [c for c in val_df.columns if c not in NON_FEATURE_COLS]

    X_val = val_df[feature_cols]
    y_val = val_df["contact"].values

    # Generate Ensemble Predictions
    p_lgbm = expert_models["lgbm"].predict_proba(X_val)[:, 1]
    p_xgb = expert_models["xgb"].predict_proba(X_val)[:, 1]
    y_pred_proba = (p_lgbm + p_xgb) / 2.0

    # Optimize Threshold
    best_threshold = 0.5
    best_mcc = -1.0
    thresholds = np.linspace(0.01, 0.99, 100)

    for thresh in thresholds:
        y_pred_bin = (y_pred_proba >= thresh).astype(int)
        score = matthews_corrcoef(y_val, y_pred_bin)
        if score > best_mcc:
            best_mcc = score
            best_threshold = thresh

    # Print Required Metric
    print(f"Final Validation Metric: {best_mcc}")

    # Save optimal threshold for InferencePipeline
    os.makedirs(os.path.join(WORKING_DIR, "models"), exist_ok=True)
    np.save(
        os.path.join(WORKING_DIR, "models/best_threshold.npy"),
        np.array([best_threshold]),
    )

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("Failure Analysis:")
    errors = np.abs(y_val - y_pred_proba)

    # Create analysis dataframe
    analysis_df = val_df[feature_cols].copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    # Filter for numeric columns just in case
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = (
        analysis_df[numeric_cols]
        .corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error Magnitude:")
    # Drop the error column itself and print top 5
    print(correlations.drop("error_magnitude", errors="ignore").head(5))

    # =========================================================================
    # 6. Submission
    # =========================================================================
    if best_mcc > 0.6865:
        InferencePipeline.run_inference(load_cached_data=True)
    else:
        print(
            f"Validation metric {best_mcc} is not higher than 0.6865. Submission generation skipped."
        )


if __name__ == "__main__":
    main()
