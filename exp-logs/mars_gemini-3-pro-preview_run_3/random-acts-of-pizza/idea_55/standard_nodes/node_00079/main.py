import os
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import setup_logging, set_seed
from library.data_loader import load_datasets
from library.feature_engineering import get_features
from library.training_engine import HybridTrainer


def main():
    # 1. Setup
    setup_logging()
    set_seed(Config.RANDOM_STATE)

    # 2. Load Data
    logging.info("Loading datasets...")
    train_df, val_df, test_df = load_datasets(load_cache=True)

    # 3. Feature Engineering
    logging.info("Extracting features...")
    # Train features (fits pipeline)
    train_feats, pipeline = get_features(train_df, "train", load_cache=True)

    # Val features (transform using fitted pipeline)
    val_feats = get_features(val_df, "val", pipeline=pipeline, load_cache=True)

    # Test features (transform using fitted pipeline)
    test_feats = get_features(test_df, "test", pipeline=pipeline, load_cache=True)

    # 4. Training
    logging.info("Initializing HybridTrainer...")
    trainer = HybridTrainer()

    # Prepare targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # Train Level 1 (Base Learners)
    # This handles CV-Bagging for volatile models and Full-Retrain for stable models
    trainer.train_stacking_layer(train_feats, y_train, val_feats, y_val)

    # Train Level 2 (Meta Learner)
    trainer.train_meta_learner()

    # 5. Validation Inference & Metric
    logging.info("Performing validation inference...")

    # Temporarily redirect submission path to avoid overwriting the actual submission file
    original_sub_path = Config.SUBMISSION_PATH
    temp_val_sub_path = os.path.join(Config.WORKING_DIR, "val_predictions_temp.csv")
    Config.SUBMISSION_PATH = temp_val_sub_path

    try:
        # Generate predictions for validation set
        val_pred_df = trainer.generate_predictions(val_feats, ids=val_df[Config.ID_COL])
        val_preds = val_pred_df[Config.TARGET_COL].values

        # Compute Metric
        score = roc_auc_score(y_val, val_preds)
        print(f"Final Validation Metric: {score}")

        # 6. Failure Analysis
        print("\n=== Failure Analysis ===")
        # Calculate absolute error
        errors = np.abs(y_val - val_preds)

        # Correlate error with numerical metadata features
        # We use val_df which contains the allowed metadata columns
        numeric_cols = val_df.select_dtypes(include=[np.number]).columns
        correlations = []

        for col in numeric_cols:
            if col == Config.TARGET_COL:
                continue

            # Get feature values, filling NaNs with 0 for correlation calculation
            feature_values = val_df[col].fillna(0).values

            # Skip constant columns
            if np.std(feature_values) == 0:
                continue

            corr = np.corrcoef(errors, feature_values)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top Feature Correlations with Prediction Error:")
        for name, corr in correlations[:5]:
            print(f"{name}: {corr:.4f}")

    finally:
        # Restore original submission path
        Config.SUBMISSION_PATH = original_sub_path
        # Cleanup temp file
        if os.path.exists(temp_val_sub_path):
            os.remove(temp_val_sub_path)

    # 7. Submission
    threshold = 0.7222984867326668
    if score > threshold:
        logging.info(
            f"Validation score {score} exceeds threshold {threshold}. Generating submission..."
        )
        trainer.generate_predictions(test_feats, ids=test_df[Config.ID_COL])
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
    else:
        logging.warning(
            f"Validation score {score} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
