import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, Timer
from library.data_loader import load_datasets
from library.features import generate_features
from library.stacking_engine import NestedStackingTrainer


def perform_failure_analysis(val_df, val_preds, target_col):
    """
    Analyzes the correlation between prediction error and numerical features.
    """
    print("\n=== Failure Analysis ===")
    y_true = val_df[target_col].values
    # Calculate absolute error
    errors = np.abs(y_true - val_preds)

    correlations = []
    # Analyze correlation with numerical columns defined in Config
    for col in Config.NUMERICAL_COLS:
        if col in val_df.columns:
            # Handle potential NaNs in raw data just for analysis
            feat_vals = val_df[col].fillna(val_df[col].median())

            # Ensure constant values don't cause warnings
            if feat_vals.nunique() > 1:
                corr, _ = pearsonr(errors, feat_vals)
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for feat, corr in correlations[:5]:
        print(f"  {feat}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    with Timer("Full Pipeline Execution"):

        # 2. Data Loading
        train_df, val_df, test_df = load_datasets(load_cached_data=True)

        # 3. Feature Generation
        # Note: generate_features handles fitting on train and transforming all splits
        train_feats, val_feats, test_feats = generate_features(
            train_df, val_df, test_df, load_cached_data=True
        )

        # 4. Training
        print("\n=== Starting Model Training ===")
        # Initialize the Stacking Engine
        trainer = NestedStackingTrainer(train_feats, train_df[Config.TARGET_COL])

        # Step A: Cross-Validation for OOF generation and Hyperparameter tuning (XGB iter)
        oof_preds = trainer.train_cv()

        # Step B: Train Level 2 Meta-Learner on OOF
        trainer.train_meta_learner(oof_preds)

        # Step C: Retrain Level 1 models on full training data
        trainer.retrain_full_models()

        # 5. Validation Inference
        print("\n=== Running Validation Inference ===")
        val_preds = trainer.predict_ensemble(val_feats)

        # Calculate Metric
        val_auc = roc_auc_score(val_df[Config.TARGET_COL], val_preds)
        print(f"Final Validation Metric: {val_auc}")

        # 6. Failure Analysis
        perform_failure_analysis(val_df, val_preds, Config.TARGET_COL)

        # 7. Submission
        threshold = 0.7085870249842536
        if val_auc > threshold:
            print(
                f"\nValidation metric ({val_auc}) > Threshold ({threshold}). Generating submission..."
            )

            # Generate Test Predictions
            test_preds = trainer.predict_ensemble(test_feats)

            # Create Submission DataFrame
            submission_df = pd.DataFrame(
                {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: test_preds}
            )

            # Ensure directory exists
            os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

            # Save
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")

            # Verify shape
            print(f"Submission shape: {submission_df.shape}")
        else:
            print(
                f"\nValidation metric ({val_auc}) <= Threshold ({threshold}). Skipping submission."
            )


if __name__ == "__main__":
    main()
