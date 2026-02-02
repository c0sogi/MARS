import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from provided library files
from library.config import Config
from library.utils import setup_logger, compute_rmse, seed_everything
from library.feature_extraction import extract_features
from library.stacking import RidgeStacker


def main():
    # 1. Setup
    logger = setup_logger("RunFile")
    seed_everything(Config.SEED)
    logger.info("Starting execution of runfile.py")

    # 2. Data Loading & Feature Extraction
    # We load features for Train, Val, and Test sets.
    # The extract_features function handles caching and GPU usage automatically.

    logger.info("--- Step 1: Feature Extraction ---")

    # Helper to load all experts for a specific split
    def load_split_features(metadata_path, mode):
        features_map = {}
        meta_data = None
        targets = None
        ids = None

        # Iterate over the 3 defined experts: clip, dinov2, convnext
        for model_name in Config.MODELS.keys():
            logger.info(f"Loading features for {model_name} [{mode}]...")
            # extract_features handles loading from cache if available (load_cached_data=True)
            f, m, t, i = extract_features(
                model_name=model_name,
                metadata_path=metadata_path,
                mode=mode,
                load_cached_data=True,
            )
            features_map[model_name] = f

            # Meta, targets, ids are consistent across experts for the same split
            # We only need to capture them once
            if meta_data is None:
                meta_data = m
                targets = t
                ids = i

        return features_map, meta_data, targets, ids

    # Load Train Set (80% of data)
    train_feats, train_meta, train_targets, _ = load_split_features(
        Config.TRAIN_METADATA_PATH, "train"
    )

    # Load Validation Set (20% of data, hold-out)
    val_feats, val_meta, val_targets, _ = load_split_features(
        Config.VAL_METADATA_PATH, "val"
    )

    # Load Test Set (Unlabeled)
    test_feats, test_meta, _, test_ids = load_split_features(
        Config.TEST_METADATA_PATH, "test"
    )

    # 3. Model Training
    logger.info("\n--- Step 2: Model Training (Stacking) ---")
    # We train the stacker on the Training set.
    # This involves:
    #   Level-0: 5-Fold CV on Train to get OOF preds and train 5 base models per expert.
    #   Level-1: Train Meta-Learner on the generated OOF preds.

    stacker = RidgeStacker()

    # Fit Level-0 Experts
    # Returns OOF predictions dataframe (N_train, N_experts)
    logger.info("Fitting Level-0 Experts...")
    oof_df = stacker.fit_level0(train_feats, train_meta, train_targets)

    # Fit Level-1 Meta-Learner
    logger.info("Fitting Level-1 Meta-Learner...")
    stacker.fit_level1(oof_df, train_targets)

    # 4. Validation & Metric
    logger.info("\n--- Step 3: Validation ---")
    # Predict on Validation set using the trained stacker
    # The stacker averages predictions from the 5 base models per expert (bagging),
    # then feeds these aggregated expert predictions to the meta-learner.
    val_preds = stacker.predict(val_feats, val_meta)

    # Compute RMSE on the hold-out set
    val_rmse = compute_rmse(val_targets, val_preds)

    # PRINT REQUIRED METRIC
    # Must be in format: "Final Validation Metric: <value>"
    print(f"Final Validation Metric: {val_rmse}")

    # 5. Failure Analysis
    logger.info("\n--- Step 4: Failure Analysis ---")
    # Calculate absolute error for each validation sample
    errors = np.abs(val_targets - val_preds)

    # Create DataFrame for analysis
    # Map binary meta features back to their names for interpretability
    analysis_df = pd.DataFrame(val_meta, columns=Config.META_FEATURES)
    analysis_df["Error"] = errors

    # Calculate correlation between Error and Metadata features
    # This helps identify which features (e.g., Blur, Occlusion) correlate with higher errors
    correlations = analysis_df.corr()["Error"].drop(["Error"])
    correlations = correlations.sort_values(ascending=False)

    logger.info("Correlation between Error Magnitude and Metadata Features:")
    print(correlations.to_string())

    # 6. Submission Generation
    logger.info("\n--- Step 5: Submission Generation ---")
    THRESHOLD = 17.163382138328082

    if val_rmse < THRESHOLD:
        logger.info(
            f"Validation RMSE ({val_rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        final_test_preds = stacker.predict(test_feats, test_meta)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_test_preds}
        )

        # Save to file
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

        # Quick verification
        saved_df = pd.read_csv(Config.SUBMISSION_PATH)
        logger.info(f"Saved file shape: {saved_df.shape}")
        logger.info(f"Head:\n{saved_df.head()}")

    else:
        logger.info(
            f"Validation RMSE ({val_rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
