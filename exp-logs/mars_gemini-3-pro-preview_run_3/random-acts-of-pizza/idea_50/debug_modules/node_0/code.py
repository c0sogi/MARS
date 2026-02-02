import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Import library components
import library.config as config
from library.utils import set_seed, Timer
from library.data_loader import load_dataset
from library.features import FeatureEngineer
from library.pipeline import HybridStackingRunner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("=== Starting Demonstration of Hex-View Stacking Ensemble ===")

    # -------------------------------------------------------------------------
    # 1. Patch Configuration for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 1] Patching configuration for speed...")

    # Reduce Cross-Validation folds
    config.N_FOLDS = 2

    # Reduce Estimators and relax constraints for Base Models
    # Random Forests
    config.LEXICAL_RF_PARAMS.update({"n_estimators": 5, "n_jobs": 1})
    config.COMMUNITY_RF_PARAMS.update({"n_estimators": 5, "n_jobs": 1})
    config.SEMANTIC_RF_PARAMS.update({"n_estimators": 5, "n_jobs": 1})

    # Boosters (XGBoost / LightGBM)
    # Disable early stopping rounds in config if set, or set to None/low value
    # Note: The pipeline code handles early stopping logic, but we reduce estimators here.
    config.SEMANTIC_XGB_PARAMS.update(
        {"n_estimators": 5, "n_jobs": 1, "early_stopping_rounds": None}
    )
    config.TEMPORAL_LGBM_PARAMS.update(
        {"n_estimators": 5, "n_jobs": 1, "early_stopping_rounds": None}
    )

    # Reduce Embedding Batch Size
    config.EMBEDDING_BATCH_SIZE = 8

    # Ensure reproducibility
    set_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading and Subsampling
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading and subsampling data...")

    # Load raw data (ignoring cache to ensure fresh load)
    train_df, val_df, test_df = load_dataset(load_cached_data=False)

    print(f"Original Train Shape: {train_df.shape}")
    print(f"Original Test Shape: {test_df.shape}")

    # Subsample to a tiny dataset for demonstration speed
    # We take 50 training samples, 20 validation, 20 test
    subset_train_size = 50
    subset_val_size = 20
    subset_test_size = 20

    train_df = train_df.iloc[:subset_train_size].copy().reset_index(drop=True)
    val_df = val_df.iloc[:subset_val_size].copy().reset_index(drop=True)
    test_df = test_df.iloc[:subset_test_size].copy().reset_index(drop=True)

    print(f"Subsampled Train Shape: {train_df.shape}")

    # Verify we have both classes in training to avoid CV errors
    unique_targets = train_df[config.TARGET_COL].nunique()
    if unique_targets < 2:
        # Fallback: Manually force class diversity if the first 50 are all same class
        # (Highly unlikely given the dataset, but robust for demo)
        train_df.loc[0, config.TARGET_COL] = 0
        train_df.loc[1, config.TARGET_COL] = 1
        print("  (Forced class diversity in subset)")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 3] Generating features...")

    fe = FeatureEngineer()

    # Generate features (force re-computation)
    train_feats, val_feats, test_feats = fe.generate_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify Feature Dictionary Structure and Shapes
    print("Verifying feature integrity...")
    expected_keys = ["lexical", "community", "semantic", "metadata"]

    for key in expected_keys:
        assert key in train_feats, f"Missing key {key} in train features"
        # Check first dimension (rows) matches dataframe
        assert train_feats[key].shape[0] == len(
            train_df
        ), f"Shape mismatch for {key}: {train_feats[key].shape[0]} vs {len(train_df)}"

    print("Feature generation successful.")

    # -------------------------------------------------------------------------
    # 4. Pipeline Execution (Stacking)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Hybrid Stacking Pipeline...")

    runner = HybridStackingRunner()

    # Prepare inputs
    train_y = train_df[config.TARGET_COL].values
    val_y = val_df[config.TARGET_COL].values
    test_ids = test_df[config.ID_COL].values

    # Execute
    # This will run CV, train base models, train meta learner, and predict
    final_preds = runner.run_stacking(
        train_feats, val_feats, test_feats, train_y, val_y, test_ids
    )

    # -------------------------------------------------------------------------
    # 5. Verification of Results
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Submission...")

    submission_path = config.SUBMISSION_FILE

    # Check file existence
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    # Load and check content
    sub_df = pd.read_csv(submission_path)

    print(f"Submission Head:\n{sub_df.head()}")

    # Assertions
    assert len(sub_df) == len(
        test_df
    ), f"Submission row count mismatch: {len(sub_df)} vs {len(test_df)}"

    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], f"Invalid columns: {sub_df.columns}"

    assert sub_df["request_id"].equals(
        test_df["request_id"]
    ), "Request IDs in submission do not match test set order."

    assert (
        sub_df["requester_received_pizza"].between(0, 1).all()
    ), "Predictions are out of probability bounds [0, 1]."

    print("\n=== Demonstration Completed Successfully ===")
