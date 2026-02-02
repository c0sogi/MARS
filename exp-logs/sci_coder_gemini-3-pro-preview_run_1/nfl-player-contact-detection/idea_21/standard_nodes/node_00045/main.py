import os
import sys
import pandas as pd
import numpy as np

# Import provided library components
import library.config as config
from library.trainer import Trainer
from library.utils import compute_mcc, get_logger, suppress_warnings

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline Execution
# -----------------------------------------------------------------------------
# To ensure the solution runs quickly within the time limit, we reduce the
# number of boosting rounds. The default of 3000 is optimized for maximum
# score but takes longer. 800 is sufficient for a strong baseline.
config.LGBM_PARAMS["n_estimators"] = 800
config.XGB_PARAMS["n_estimators"] = 800

# Ensure silent execution
config.LGBM_PARAMS["verbose"] = -1
config.XGB_PARAMS["verbosity"] = 0


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    # Suppress warnings for cleaner output
    suppress_warnings()
    logger = get_logger("runfile")

    logger.info("Starting Vector-Decomposed Spectral Mining Ensemble Pipeline...")

    # Initialize the Trainer which orchestrates DataFactory and ModelFactory
    trainer = Trainer()

    # -------------------------------------------------------------------------
    # 1. Training Pipeline
    # -------------------------------------------------------------------------
    # This step:
    # - Generates/Loads features for Train/Val
    # - Trains Scout models (LGBM/XGB) on balanced data
    # - Mines Hard Negatives (Union of Scout errors)
    # - Trains Expert models on Hard Negative enriched data
    # - Optimizes the decision threshold
    ensemble, best_thresh = trainer.run_pipeline(
        load_cached_features=True, load_cached_mining=True
    )

    # -------------------------------------------------------------------------
    # 2. Validation & Metrics
    # -------------------------------------------------------------------------
    logger.info("--- Final Validation Assessment ---")

    # Load the full validation set
    # Note: We load directly via DataFactory to ensure we have the raw data for analysis
    df_val = trainer.data_factory.load_features(mode="val", load_cached_data=True)
    X_val, y_val = trainer.data_factory.get_validation_data(df_val)

    # Perform Inference
    # The ensemble averages probabilities from the Expert LGBM and Expert XGB
    y_prob = ensemble.predict(X_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    # Compute Matthews Correlation Coefficient
    mcc = compute_mcc(y_val, y_pred)

    # Print the required metric string
    print(f"Final Validation Metric: {mcc}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("--- Performing Failure Analysis ---")

    # Calculate Error Magnitude: |Ground Truth - Predicted Probability|
    # High error magnitude indicates confident incorrect predictions
    errors = np.abs(y_val - y_prob)

    # Create an analysis dataframe combining features and errors
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation between features and error magnitude
    # This reveals which features are systematically associated with model failure
    correlations = (
        analysis_df.corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\nFailure Analysis - Top 10 Feature Correlations with Error Magnitude:")
    print(correlations.head(10))

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.6865

    if mcc > THRESHOLD_SCORE:
        logger.info(
            f"Validation MCC ({mcc:.5f}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Load Test Data
        df_test_features = trainer.data_factory.load_features(
            mode="test", load_cached_data=True
        )
        X_test = trainer.data_factory.get_test_data(df_test_features)

        # Generate Predictions
        test_probs = ensemble.predict(X_test)
        test_preds = (test_probs >= best_thresh).astype(int)

        # Create Prediction DataFrame
        # We assume df_test_features preserves the contact_id from metadata
        sub_df = pd.DataFrame(
            {"contact_id": df_test_features["contact_id"], "contact": test_preds}
        )

        # Load Sample Submission Template
        # This is crucial to ensure we output exactly the rows expected by the leaderboard
        sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
        sample_sub = pd.read_csv(sample_sub_path)

        # Merge predictions onto the sample submission
        # Left merge on sample_sub ensures we keep the correct order and row count
        final_sub = sample_sub[["contact_id"]].merge(
            sub_df, on="contact_id", how="left"
        )

        # Fill any missing predictions with 0 (No Contact) as a fallback
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

        # Save Submission
        final_sub.to_csv(config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission successfully saved to {config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation MCC ({mcc:.5f}) did not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
