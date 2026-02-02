import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib

# Import library modules
from library.config import Config
from library.utils import setup_logger, seed_everything, compute_mcc, garbage_collection
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.mining import ScoutMiner
from library.training import ExpertTrainer
from library.inference import InferencePipeline
from library.model_factory import ModelFactory


def main():
    # -------------------------------------------------------------------------
    # 0. Setup and Configuration
    # -------------------------------------------------------------------------
    logger = setup_logger("RunFile")
    seed_everything(Config.SEED)

    # Check for GPU and configure models accordingly
    if torch.cuda.is_available():
        logger.info("GPU detected. Configuring models for GPU acceleration.")
        # Update LightGBM params
        Config.SCOUT_LGBM_PARAMS["device"] = "gpu"
        Config.EXPERT_LGBM_PARAMS["device"] = "gpu"
        # Update XGBoost params
        Config.EXPERT_XGB_PARAMS["device"] = "cuda"
        Config.EXPERT_XGB_PARAMS["tree_method"] = "hist"
    else:
        logger.info("No GPU detected. Using CPU.")

    # Optimization for Fast Baseline: Reduce estimators to ensure < 2h runtime
    # While maintaining enough capacity to learn
    Config.SCOUT_LGBM_PARAMS["n_estimators"] = 500
    Config.EXPERT_LGBM_PARAMS["n_estimators"] = 500
    Config.EXPERT_XGB_PARAMS["n_estimators"] = 500

    # -------------------------------------------------------------------------
    # 1. Data Loading & Tier 1 Feature Generation (for Mining)
    # -------------------------------------------------------------------------
    logger.info("--- Phase 1: Data Loading & Tier 1 Feature Generation ---")
    loader = DataLoader(debug=Config.DEBUG)
    generator = FeatureGenerator()

    # Load Train Data
    merged_train = loader.get_merged_data(split="train", load_cached_data=True)
    tracking_train = loader.load_tracking(split="train")

    # Generate Tier 1 Features for Train (Full Dataset)
    df_train_tier1 = generator.generate(
        merged_train, tracking_train, tier=1, split="train", load_cached_data=True
    )

    # Load Validation Data
    merged_val = loader.get_merged_data(split="val", load_cached_data=True)
    tracking_val = loader.load_tracking(split="val")

    # Generate Tier 1 Features for Validation
    df_val_tier1 = generator.generate(
        merged_val, tracking_val, tier=1, split="val", load_cached_data=True
    )

    # Cleanup raw data to free memory
    del merged_train, tracking_train, merged_val, tracking_val
    garbage_collection()

    # -------------------------------------------------------------------------
    # 2. Mining (Scout)
    # -------------------------------------------------------------------------
    logger.info("--- Phase 2: Scout Mining ---")
    miner = ScoutMiner()
    # Execute mining to identify Positives and Hard Negatives
    mined_indices = miner.execute(df_train_tier1, df_val_tier1, load_cached_data=True)

    # Cleanup Tier 1 data
    del df_train_tier1, df_val_tier1
    garbage_collection()

    # -------------------------------------------------------------------------
    # 3. Expert Training
    # -------------------------------------------------------------------------
    logger.info("--- Phase 3: Expert Training ---")
    trainer = ExpertTrainer()
    # Trains LGBM and XGB on the mined subset using Tier 2 features
    trainer.train(mined_indices, load_cached_data=True)

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("--- Phase 4: Validation & Failure Analysis ---")

    # We need to re-generate Tier 2 features for Validation to perform the final assessment
    # (The trainer generates them internally but cleans them up)
    loader = DataLoader(debug=Config.DEBUG)
    merged_val = loader.get_merged_data(split="val", load_cached_data=True)
    tracking_val = loader.load_tracking(split="val")

    df_val_tier2 = generator.generate(
        merged_val, tracking_val, tier=2, split="val", load_cached_data=True
    )

    # Load Trained Models and Threshold
    lgbm_model = ModelFactory.create_model(stage="expert", model_type="lgbm")
    lgbm_model.load(trainer.lgbm_path)

    xgb_model = ModelFactory.create_model(stage="expert", model_type="xgb")
    xgb_model.load(trainer.xgb_path)

    threshold = joblib.load(trainer.threshold_path)

    # Prepare Validation Features
    feature_cols = [c for c in Config.TIER2_FEATURES if c in df_val_tier2.columns]
    X_val = df_val_tier2[feature_cols]
    y_val = df_val_tier2["contact"].values

    # Ensemble Prediction
    # Note: Models are in sklearn wrapper, predict_proba is efficient and doesn't compute gradients
    p_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]
    p_ensemble = (p_lgbm + p_xgb) / 2.0

    preds = (p_ensemble >= threshold).astype(int)

    # Compute Metric
    final_mcc = compute_mcc(y_val, preds)
    print(f"Final Validation Metric: {final_mcc}")

    # Failure Analysis: Correlation of Error with Features
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_val - p_ensemble)

    # Select key features for analysis
    analysis_cols = [
        "distance",
        "speed_diff",
        "acc_diff",
        "orient_diff",
        "spatial_density",
        "cluster_speed",
    ]
    # Add lags if they exist
    analysis_cols += [
        f"{c}_lag1" for c in ["distance", "speed_diff"] if f"{c}_lag1" in X_val.columns
    ]

    correlations = {}
    for col in analysis_cols:
        if col in X_val.columns:
            # Handle potential NaNs just in case, though preprocessing should have handled them
            valid_mask = ~np.isnan(X_val[col])
            if valid_mask.sum() > 0:
                corr = np.corrcoef(errors[valid_mask], X_val.loc[valid_mask, col])[0, 1]
                correlations[col] = corr
            else:
                correlations[col] = 0.0

    logger.info("Correlation of Prediction Error with Input Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        logger.info(f"{feat}: {corr:.4f}")

    # Cleanup Validation Data
    del df_val_tier2, X_val, p_lgbm, p_xgb, p_ensemble
    garbage_collection()

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    TARGET_SCORE = 0.6746827603428585

    if final_mcc > TARGET_SCORE:
        logger.info(
            f"Validation score ({final_mcc:.6f}) exceeds threshold ({TARGET_SCORE:.6f}). Proceeding to Submission."
        )
        pipeline = InferencePipeline()
        pipeline.run(load_cached_data=True)
    else:
        logger.warning(
            f"Validation score ({final_mcc:.6f}) did not meet threshold ({TARGET_SCORE:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
