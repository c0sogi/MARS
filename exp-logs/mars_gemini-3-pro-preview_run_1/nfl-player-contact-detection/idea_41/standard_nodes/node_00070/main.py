import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, matthews_corrcoef

# Import from library
from library.config import CACHE_DIR, SEED, LGBM_PARAMS, XGB_PARAMS, SUBMISSION_FILE
from library.utils import (
    setup_logger,
    seed_everything,
    optimize_threshold,
    CacheManager,
)
from library.data_processing import DataProcessor
from library.feature_engineering import FeatureEngineer
from library.mining_strategy import MiningStrategy
from library.model_definitions import LGBMExpert, XGBExpert
from library.inference import InferencePipeline


def main():
    # 1. Setup
    logger = setup_logger(os.path.join(os.getcwd(), "logs", "runfile.log"))
    seed_everything(SEED)
    logger.info("Starting Fast Baseline Run...")

    # Configuration for Fast Baseline
    TRAIN_SAMPLE_SIZE = 100000  # Limit training data for speed
    FAST_N_ESTIMATORS = 500  # Limit trees for speed

    # Override params
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params["n_estimators"] = FAST_N_ESTIMATORS

    xgb_params = XGB_PARAMS.copy()
    xgb_params["n_estimators"] = FAST_N_ESTIMATORS

    # Directories
    model_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # =========================================================================
    # 2. Data Loading & Feature Engineering
    # =========================================================================
    processor = DataProcessor(logger=logger)
    engineer = FeatureEngineer(logger=logger)

    logger.info(f"Loading Training Metadata (Sampled: {TRAIN_SAMPLE_SIZE})...")
    df_meta_train = processor.load_metadata(
        split="train", sample_size=TRAIN_SAMPLE_SIZE
    )

    logger.info("Loading Validation Metadata (Full)...")
    df_meta_val = processor.load_metadata(split="val")  # Must use full for valid metric

    logger.info("Loading Tracking Data...")
    df_tracking = processor.load_tracking(split="train")

    logger.info("Generating Training Features...")
    # Note: We disable loading from cache for train to ensure we use the sampled version
    # unless we carefully manage cache keys. For safety in this baseline, we re-compute
    # or rely on the engineer's internal caching if it handles sampling (it doesn't by default key).
    # We will force re-compute for the sampled train set to be safe.
    df_features_train = engineer.create_features(
        df_meta_train,
        df_tracking,
        split="train",
        load_cached_data=False,
        save_output=False,
    )

    logger.info("Generating Validation Features...")
    df_features_val = engineer.create_features(
        df_meta_val, df_tracking, split="val", load_cached_data=True, save_output=True
    )

    # Prepare X and y
    exclude_cols = [
        "contact_id",
        "contact",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "nfl_player_id_2_numeric",
    ]
    feature_cols = [c for c in df_features_train.columns if c not in exclude_cols]

    X_train = df_features_train[feature_cols]
    y_train = df_features_train["contact"]

    X_val = df_features_val[feature_cols]
    y_val = df_features_val["contact"]

    # Cleanup
    del df_meta_train, df_meta_val, df_tracking, df_features_train, df_features_val
    gc.collect()

    # =========================================================================
    # 3. Mining Strategy
    # =========================================================================
    miner = MiningStrategy(logger=logger)

    # Train Scouts
    logger.info("Training Scouts...")
    scout_lgbm, scout_xgb = miner.train_scouts(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=feature_cols,
        load_cached_models=False,  # Force retrain on sampled data
    )

    # Mine Hard Negatives
    logger.info("Mining Hard Negatives...")
    hard_neg_indices = miner.mine_hard_negatives(
        scout_lgbm, scout_xgb, X_train, y_train, load_cached_indices=False
    )

    # Construct Expert Dataset
    X_expert, y_expert = miner.construct_expert_dataset(
        X_train, y_train, hard_neg_indices
    )

    # =========================================================================
    # 4. Train Expert Models
    # =========================================================================
    logger.info("Training Expert Models...")

    expert_lgbm = LGBMExpert(params=lgbm_params, logger=logger)
    expert_lgbm.fit(X_expert, y_expert, X_val, y_val, feature_names=feature_cols)
    expert_lgbm.save(os.path.join(model_dir, "expert_lgbm.joblib"))

    expert_xgb = XGBExpert(params=xgb_params, logger=logger)
    expert_xgb.fit(X_expert, y_expert, X_val, y_val, feature_names=feature_cols)
    expert_xgb.save(os.path.join(model_dir, "expert_xgb.joblib"))

    # =========================================================================
    # 5. Validation & Threshold Optimization
    # =========================================================================
    logger.info("Running Validation...")

    preds_lgbm = expert_lgbm.predict_proba(X_val)
    preds_xgb = expert_xgb.predict_proba(X_val)
    preds_ensemble = (preds_lgbm + preds_xgb) / 2.0

    best_threshold, best_mcc = optimize_threshold(y_val.values, preds_ensemble)

    # Save threshold
    np.save(os.path.join(model_dir, "best_threshold.npy"), np.array([best_threshold]))

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {best_mcc}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    logger.info("Performing Failure Analysis...")

    # Calculate error magnitude (abs diff)
    errors = np.abs(y_val.values - preds_ensemble)

    # Calculate correlation with features
    correlations = {}
    for col in X_val.columns:
        # Simple correlation
        if pd.api.types.is_numeric_dtype(X_val[col]):
            corr = np.corrcoef(X_val[col].fillna(0), errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nFailure Analysis - Top Feature Correlations with Error:")
    for name, val in sorted_corr[:10]:
        print(f"{name}: {val:.4f}")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    THRESHOLD_SCORE = 0.6865

    if best_mcc > THRESHOLD_SCORE:
        logger.info(
            f"Validation score {best_mcc} > {THRESHOLD_SCORE}. Generating submission..."
        )

        # Initialize Inference Pipeline
        # It loads models from the model_dir we saved to
        inference_pipe = InferencePipeline(logger=logger)

        # Run Inference
        inference_pipe.run_inference(load_cached_data=True)

        if os.path.exists(SUBMISSION_FILE):
            logger.info(f"Submission successfully generated at {SUBMISSION_FILE}")
        else:
            logger.error("Submission file not found after inference.")
    else:
        logger.info(
            f"Validation score {best_mcc} <= {THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
