import pandas as pd
import numpy as np
import gc
import sys
from library.config import Config
from library.pipeline import IHNMEPipeline
from library.utils import seed_everything, compute_mcc


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # --------------------------------------------------------------------------
    # Reduce estimators to ensure the code completes within the 2-hour limit
    Config.LGBM_SCOUT_PARAMS["n_estimators"] = 200
    Config.LGBM_EXPERT_PARAMS["n_estimators"] = 200
    Config.XGB_EXPERT_PARAMS["n_estimators"] = 200

    # Aggressive early stopping
    Config.EARLY_STOPPING_ROUNDS = 20

    # --------------------------------------------------------------------------
    # 2. Initialization
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    pipeline = IHNMEPipeline()

    # --------------------------------------------------------------------------
    # 3. Data Loading & Subsampling
    # --------------------------------------------------------------------------
    # Load full datasets (utilizing cache if available)
    train_df = pipeline.prepare_data(split="train", load_cached_data=True)
    val_df = pipeline.prepare_data(split="val", load_cached_data=True)

    # Subsample training data for fast baseline execution
    # The full dataset is ~3.4M rows, which is too slow for a quick check.
    # Increased to 600k to improve Hard Negative Mining coverage (Cite solution_lesson_node_00019)
    MAX_TRAIN_SAMPLES = 600000
    if len(train_df) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_df)} to {MAX_TRAIN_SAMPLES}..."
        )
        # We use a random sample. The pipeline's internal sampler handles class imbalance later.
        train_df = train_df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)

    feature_cols = pipeline._get_feature_cols(train_df)

    # --------------------------------------------------------------------------
    # 4. Phase 1: Scout Training
    # --------------------------------------------------------------------------
    scout_model = pipeline.run_phase_1_scout(train_df, val_df, feature_cols)

    # --------------------------------------------------------------------------
    # 5. Phase 2: Hard Negative Mining
    # --------------------------------------------------------------------------
    # Mine hard negatives from the (subsampled) training set
    hard_negs_df = pipeline.run_phase_2_mining(scout_model, train_df, feature_cols)

    # Clean up Scout model to save memory
    del scout_model
    gc.collect()

    # --------------------------------------------------------------------------
    # 6. Phase 3: Expert Training
    # --------------------------------------------------------------------------
    expert_models = pipeline.run_phase_3_expert(
        train_df, hard_negs_df, val_df, feature_cols
    )

    # --------------------------------------------------------------------------
    # 7. Validation & Threshold Optimization
    # --------------------------------------------------------------------------
    print("--- Validation ---")
    X_val = val_df[feature_cols]
    y_val = val_df["contact"].values

    # Ensemble Prediction (Average)
    preds = np.zeros(len(X_val))
    for model in expert_models:
        preds += model.predict(X_val)
    preds /= len(expert_models)

    # Grid Search for Threshold
    best_threshold = 0.5
    best_mcc = -1.0
    thresholds = np.arange(0.1, 0.95, 0.01)

    for thresh in thresholds:
        y_pred_bin = (preds > thresh).astype(int)
        mcc = compute_mcc(y_val, y_pred_bin)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {best_mcc}")

    # --------------------------------------------------------------------------
    # 8. Failure Analysis
    # --------------------------------------------------------------------------
    print("--- Failure Analysis ---")
    # Calculate error magnitude (confident wrong predictions have high error)
    errors = np.abs(y_val - preds)

    correlations = {}
    for col in feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(X_val[col]):
            # Filter out NaNs
            valid_mask = ~np.isnan(X_val[col]) & ~np.isnan(errors)
            if valid_mask.sum() > 10:
                corr = np.corrcoef(X_val.loc[valid_mask, col], errors[valid_mask])[0, 1]
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    # --------------------------------------------------------------------------
    # 9. Submission
    # --------------------------------------------------------------------------
    TARGET_METRIC = 0.658992501127342

    if best_mcc > TARGET_METRIC:
        print(
            f"Validation metric {best_mcc} > {TARGET_METRIC}. Generating submission..."
        )
        pipeline.generate_submission(expert_models, best_threshold)
    else:
        print(
            f"Validation metric {best_mcc} did not meet threshold {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
