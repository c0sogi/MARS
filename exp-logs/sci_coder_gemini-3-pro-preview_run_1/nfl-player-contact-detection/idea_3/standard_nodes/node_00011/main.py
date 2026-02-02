import pandas as pd
import numpy as np
import gc
import sys
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import seed_everything, print_metric
from library.data_pipeline import DataPipeline
from library.model_interaction import InteractionGBM
from library.model_impact import ImpactTrainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # Enforce reproducibility
    seed_everything(Config.SEED)

    # Adjust hyperparameters for a fast run (under 2 hours)
    # Reduce LightGBM rounds but ensure enough to learn
    Config.LGBM_TRAIN_PARAMS["num_boost_round"] = 1000
    Config.LGBM_TRAIN_PARAMS["early_stopping_rounds"] = 50

    print("Configuration initialized. Starting Unified Pipeline...")

    # Initialize Data Pipeline
    pipeline = DataPipeline(Config)

    # -------------------------------------------------------------------------
    # 2. Unified Model Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("UNIFIED MODEL TRAINING")
    print("=" * 40)

    # Load Unified Data
    # Increase subsample size to 1M to maximize info density (Cite Lesson 00003)
    train_df, feats = pipeline.get_unified_data("train", subsample_size=1000000)
    val_df, _ = pipeline.get_unified_data("val")

    # Train LightGBM (Unified)
    model = InteractionGBM(Config)
    model.train(train_df, val_df, feats)

    # Generate Validation Predictions
    val_preds = model.predict(val_df, feats)
    val_df["pred_prob"] = val_preds

    # Cleanup to save memory
    del train_df
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Global Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("GLOBAL EVALUATION")
    print("=" * 40)

    # Calculate MCC on the full validation set
    y_true = val_df["contact"].values
    y_prob = val_df["pred_prob"].values
    y_pred = (y_prob > 0.5).astype(int)

    final_mcc = matthews_corrcoef(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate error magnitude
    val_df["error"] = np.abs(val_df["contact"] - val_df["pred_prob"])

    print("Global Error Stats:")
    print(val_df["error"].describe())

    # Error Correlation Analysis
    print("\nError Correlation with Features:")
    corr_matrix = val_df[feats + ["error"]].corr()
    error_corrs = corr_matrix["error"].drop("error").sort_values(ascending=False)

    print("Top 5 Features associated with Error:")
    print(error_corrs.head(5))

    # -------------------------------------------------------------------------
    # 6. Submission Logic
    # -------------------------------------------------------------------------
    THRESHOLD = 0.5979349867102601

    if final_mcc > THRESHOLD:
        print("\n" + "=" * 40)
        print("GENERATING SUBMISSION")
        print("=" * 40)

        # 1. Inference (Test)
        print("Predicting Test Set...")
        test_df, _ = pipeline.get_unified_data("test")
        test_preds_prob = model.predict(test_df, feats)
        test_df["contact"] = (test_preds_prob > 0.5).astype(int)

        # 3. Merge and Format
        submission_preds = test_df[["contact_id", "contact"]]

        # Load sample submission to ensure correct order and completeness
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into sample submission structure
        # Left join ensures we keep all rows from sample_sub
        final_submission = sample_sub[["contact_id"]].merge(
            submission_preds, on="contact_id", how="left"
        )

        # Fill missing (if any) with 0 and ensure int type
        final_submission["contact"] = final_submission["contact"].fillna(0).astype(int)

        # Save
        final_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {final_submission.shape}")

    else:
        print(
            f"\nValidation Metric ({final_mcc}) did not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
