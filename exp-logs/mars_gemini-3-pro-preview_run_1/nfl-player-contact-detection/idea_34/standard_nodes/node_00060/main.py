import os
import sys
import numpy as np
import pandas as pd
import logging

# Import provided library modules
from library import config, utils, trainer

# =============================================================================
# 1. CONFIGURATION OVERRIDE FOR FAST BASELINE & GPU
# =============================================================================
# Reduce estimators for speed
config.NUM_ESTIMATORS = 100
config.EARLY_STOPPING_ROUNDS = 20

# Enable GPU acceleration where possible
# Note: We modify the dictionaries in the config module directly before they are used by ModelFactory
config.LGBM_PARAMS["device"] = "gpu"
config.XGB_PARAMS["tree_method"] = "gpu_hist"
# CatBoost GPU support
config.CAT_PARAMS["task_type"] = "GPU"


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # Set seeds for reproducibility
    utils.seed_everything(config.SEED)

    # Initialize Trainer
    ct = trainer.CurriculumTrainer()

    # -------------------------------------------------------------------------
    # 2. DATA LOADING & SUBSAMPLING
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # Load cached data if available to save time
    df_train = ct.loader.prepare_dataset(split="train", load_cached_data=True)
    df_val = ct.loader.prepare_dataset(split="val", load_cached_data=True)

    # Fast Baseline: Subsample training data to ensure execution within time limit
    # We keep the validation set intact for accurate metric calculation
    MAX_TRAIN_SAMPLES = 100000
    if len(df_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(df_train)} to {MAX_TRAIN_SAMPLES}..."
        )
        df_train = df_train.sample(
            n=MAX_TRAIN_SAMPLES, random_state=config.SEED
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 3. TRAINING PIPELINE
    # -------------------------------------------------------------------------
    # Phase 1: Train Scouts
    # Scouts are trained on a balanced subset to find candidate hard negatives
    ct.train_scouts(df_train, df_val)

    # Phase 2: Mine Hard Negatives
    # Use Scouts to find negatives in the training set that look like positives
    hard_neg_indices = ct.mine_hard_negatives(df_train)

    # Phase 3: Train Experts
    # Experts are trained on Positives + Hard Negatives + Random Anchors
    ct.train_experts(df_train, hard_neg_indices, df_val)

    # -------------------------------------------------------------------------
    # 4. VALIDATION & METRIC CALCULATION
    # -------------------------------------------------------------------------
    print("Evaluating Ensemble on Validation Set...")
    X_val = df_val[ct.feature_cols]
    y_val = df_val[ct.target_col].values

    # Generate Ensemble Probabilities
    ensemble_probs = np.zeros(len(X_val))
    for m_type, model in ct.experts.items():
        probs = ct.factory.predict_proba(model, X_val)
        ensemble_probs += probs

    if len(ct.experts) > 0:
        ensemble_probs /= len(ct.experts)

    # Optimize Threshold for MCC
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_mcc = -1.0
    best_th = 0.5

    for th in thresholds:
        preds = (ensemble_probs >= th).astype(int)
        mcc = utils.calc_mcc(y_val, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_th = th

    # Save the best threshold for inference usage
    ct.best_threshold = best_th
    np.save(os.path.join(ct.models_dir, "best_threshold.npy"), np.array([best_th]))

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {best_mcc}")

    # -------------------------------------------------------------------------
    # 5. FAILURE ANALYSIS
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    # Calculate Error Magnitude
    errors = np.abs(y_val - ensemble_probs)

    # Create a temporary dataframe for correlation analysis
    # We analyze correlation between features and the error magnitude
    analysis_df = X_val.copy()
    analysis_df["__error__"] = errors

    # Compute correlations
    correlations = analysis_df.corr()["__error__"].drop("__error__")

    # Identify top features associated with error
    top_corrs = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 Features correlated with Prediction Error:")
    print(top_corrs)

    # -------------------------------------------------------------------------
    # 6. CONDITIONAL SUBMISSION
    # -------------------------------------------------------------------------
    TARGET_METRIC = 0.6865

    if best_mcc > TARGET_METRIC:
        print(
            f"\nValidation Metric ({best_mcc}) > {TARGET_METRIC}. Generating Submission..."
        )
        # predict_test handles loading test data, feature generation, and saving submission
        ct.predict_test(load_cached_data=True)
    else:
        print(
            f"\nValidation Metric ({best_mcc}) <= {TARGET_METRIC}. Skipping Submission."
        )


if __name__ == "__main__":
    main()
