import sys
import os
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

# Import from provided library files
import library.config as config
from library.utils import setup_logger, seed_everything
from library.trainer import CascadeTrainer
from library.data_loader import DataLoader
from library.features import FeatureFactory


def main():
    # 1. Initialization and Setup
    seed_everything(config.SEED)
    logger = setup_logger()

    # GPU Optimization: Detect GPU and update XGBoost parameters
    if torch.cuda.is_available():
        logger.info("GPU detected. Updating XGBoost configuration to use CUDA.")
        # Modify the configuration dictionary in place before model instantiation
        config.XGB_EXPERT_PARAMS["tree_method"] = "hist"
        config.XGB_EXPERT_PARAMS["device"] = "cuda"

    # 2. Instantiate Pipeline Components
    trainer = CascadeTrainer()
    loader = DataLoader()
    factory = FeatureFactory()

    # 3. Data Loading
    # Constraint: Limit training data to ensure fast execution (Fast Baseline)
    # Cite solution_lesson_node_00021: Maximize data volume in mining phase.
    # Increasing from 500k to 800k to improve hard negative yield.
    TRAIN_LIMIT = 800000
    logger.info(f"Loading Training Data (Limited to {TRAIN_LIMIT} rows)...")
    df_train = loader.prepare_base_table(mode="train", n_rows=TRAIN_LIMIT)

    # Constraint: Must use the entire hold-out validation set for metric calculation
    logger.info("Loading Validation Data (Full)...")
    df_val = loader.prepare_base_table(mode="val")

    # 4. Phase 1: Train Scout Model
    # Use a smaller subset for the scout to keep this phase negligible in time
    SCOUT_LIMIT = 150000
    logger.info(f"Phase 1: Training Scout on {SCOUT_LIMIT} rows...")
    scout = trainer.train_scout(df_train, n_rows=SCOUT_LIMIT)

    # 5. Phase 2: Mine Hard Negatives
    # Run scout inference on the loaded training set to find hard examples
    logger.info("Phase 2: Mining Hard Negatives...")
    mining_mask = trainer.mine_hard_negatives(df_train, scout)

    # 6. Phase 3: Train Expert Ensemble
    # Trains LGBM and XGB on mined data, optimizes threshold on df_val
    logger.info("Phase 3: Training Expert Ensemble...")
    trainer.train_experts(df_train, mining_mask, df_val)

    # 7. Validation & Failure Analysis
    logger.info("Performing Final Validation & Failure Analysis...")

    # Compute Tier 2 features for the full validation set
    # (These should be cached by the trainer, so this is fast)
    X_val = factory.compute_tier2_features(df_val)
    y_val = df_val["contact"].values

    # Retrieve trained models
    lgbm_model = trainer.models["expert_lgbm"]
    xgb_model = trainer.models["expert_xgb"]

    # Run Inference on Validation Set
    p_lgbm = lgbm_model.predict_proba(X_val)
    p_xgb = xgb_model.predict_proba(X_val)
    p_ens = 0.5 * p_lgbm + 0.5 * p_xgb

    # Apply the optimized threshold
    best_thresh = trainer.best_threshold
    preds = (p_ens >= best_thresh).astype(int)

    # Calculate Metric
    mcc = matthews_corrcoef(y_val, preds)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {mcc}")

    # Failure Analysis: Correlation of Error Magnitude with Features
    logger.info("Calculating Feature correlations with Error Magnitude...")
    errors = np.abs(y_val - p_ens)

    # Calculate correlation between features and the error vector
    # We use corrwith for efficient column-wise correlation
    corr_series = X_val.corrwith(pd.Series(errors, index=X_val.index))
    corr_series = corr_series.abs().sort_values(ascending=False)

    logger.info("Top 10 Features associated with Model Error:")
    logger.info(corr_series.head(10))

    # Cleanup memory before potential inference
    del X_val, p_lgbm, p_xgb, p_ens, errors, corr_series
    gc.collect()

    # 8. Submission Logic
    SUBMISSION_THRESHOLD = 0.6746827603428585

    if mcc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation Metric ({mcc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Proceeding to Submission."
        )

        # Load Test Data
        df_test = loader.prepare_base_table(mode="test")

        # Generate Submission
        trainer.generate_submission(df_test)
        logger.info("Submission generated successfully.")
    else:
        logger.info(
            f"Validation Metric ({mcc}) does not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
