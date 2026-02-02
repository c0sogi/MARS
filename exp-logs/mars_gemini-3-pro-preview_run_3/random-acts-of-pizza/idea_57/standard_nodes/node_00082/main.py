import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger, save_submission, print_metric
from library.data_factory import prepare_datasets
from library.feature_pipeline import FeatureManager
from library.stacking_engine import HybridEnsembleTrainer


def main():
    # 1. Setup
    logger = setup_logger("main")
    set_seed(Config.RANDOM_STATE)

    logger.info("Starting Runfile Execution...")

    # 2. Data Loading
    # prepare_datasets loads train and val, merges them into union_train, and loads test
    # This supports the architecture's requirement for a larger training corpus via CV
    union_train_df, test_df = prepare_datasets(load_cached_data=True)

    # Load the specific validation set for final reporting requirements
    val_df = pd.read_parquet(Config.VAL_PATH)
    logger.info(f"Loaded Hold-out Validation Set: {val_df.shape}")

    # 3. Feature Processing
    feature_manager = FeatureManager()
    # This computes features for union_train and test
    feature_data = feature_manager.process_features(
        union_train_df, test_df, load_cached_data=True
    )

    # 4. Training & Stacking
    trainer = HybridEnsembleTrainer(feature_data, union_train_df, test_df)

    # Step 4.1: Train Level 1 (Base Learners) with CV
    # This generates OOF predictions for the entire union dataset (including the validation subset)
    trainer.train_level_1()

    # Step 4.2: Train Level 2 (Meta Learner)
    # Trains on the OOF predictions
    trainer.train_meta_learner()

    # Step 4.3: Retrain Stable Models on Full Union Data
    # Maximizes data usage for stable learners (RF, LR) for the final Test Inference
    trainer.retrain_stable_models()

    # 5. Validation & Failure Analysis
    logger.info("Performing Final Validation on Hold-out Set...")

    # Strategy:
    # Since val_df is part of union_train_df, we extract the OOF predictions
    # corresponding to the validation IDs. These OOF predictions come from
    # models that did NOT see these specific rows during Level 1 training (due to CV).
    # We then feed these OOFs to the Meta-Learner.

    # Get OOF predictions from trainer (indexed by row logic, but we have IDs)
    oof_df = trainer.oof_df.copy()

    # Merge with val_df to filter rows and get targets
    # oof_df has columns: request_id, [model_keys...]
    val_eval_df = val_df.merge(oof_df, on=Config.ID_COL, how="inner")

    if len(val_eval_df) == 0:
        logger.error(
            "Validation IDs not found in OOF DataFrame. Check data merging logic."
        )
        return

    # Prepare Meta-Features for Validation
    model_keys = list(Config.MODEL_CONFIGS.keys())
    X_val_meta = val_eval_df[model_keys].values
    y_val_true = val_eval_df[Config.TARGET_COL].values

    # Predict using Meta-Learner
    # Note: We use the trained meta-learner to combine the OOF scores.
    val_preds = trainer.meta_learner.predict_proba(X_val_meta)[:, 1]

    # Compute Metric
    final_auc = roc_auc_score(y_val_true, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    val_eval_df["prediction"] = val_preds
    val_eval_df["error"] = np.abs(
        val_eval_df[Config.TARGET_COL] - val_eval_df["prediction"]
    )

    # Correlate error with numerical features in val_df
    # We use the raw numerical columns present in val_df
    numerical_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = []

    for col in numerical_cols:
        if col in [Config.TARGET_COL, "prediction", "error"]:
            continue
        # Handle NaNs for correlation calculation
        series = val_eval_df[col].fillna(0)
        if series.nunique() > 1:
            corr, _ = pearsonr(series, val_eval_df["error"])
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.7222984867326668

    if final_auc > THRESHOLD:
        logger.info(
            f"Metric ({final_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Generate Test Predictions
        # This uses the Hybrid Inference strategy:
        # - Volatile Models: Average of 5 CV-Fold Models (CV-Bagging)
        # - Stable Models: Single Retrained Model
        test_preds = trainer.predict()

        # Save
        save_submission(test_df[Config.ID_COL].values, test_preds)
    else:
        logger.warning(
            f"Metric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
