import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import provided library modules
from library import (
    config,
    utils,
    data_processing,
    feature_engineering,
    models,
    training_pipeline,
    inference,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Set seeds
    utils.seed_everything(config.SEED)

    # Override Config for Speed
    config.N_ESTIMATORS = 10
    config.EARLY_STOPPING_ROUNDS = 5
    config.VERBOSE_EVAL = -1  # Silent

    # Override Model Params to be silent and fast
    config.LGBM_PARAMS["verbosity"] = -1
    config.LGBM_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["verbosity"] = 0
    config.XGB_PARAMS["n_estimators"] = 10

    # Ensure working directories are clean/ready
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 2] Generating features (Sample Size: 500)...")

    # Generate Training Features (Small Subset)
    # We use load_cached_data=False to force computation for demonstration
    df_train = feature_engineering.generate_train_features(
        debug=True, sample_size=500, load_cached_data=False
    )

    # Generate Validation Features (Small Subset)
    df_val = feature_engineering.generate_val_features(
        debug=True, sample_size=200, load_cached_data=False
    )

    # Verification
    assert not df_train.empty, "Training dataframe is empty!"
    assert not df_val.empty, "Validation dataframe is empty!"
    assert (
        "contact" in df_train.columns
    ), "Target column 'contact' missing from training data."

    # Check for specific feature engineering artifacts (e.g., lag features)
    expected_col = "dist_t+0"
    assert (
        expected_col in df_train.columns
    ), f"Expected feature '{expected_col}' not found."

    print(f"  Train Shape: {df_train.shape}")
    print(f"  Val Shape:   {df_val.shape}")

    # Prepare X and y
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    X_train_full = df_train[feature_cols]
    y_train_full = df_train["contact"]
    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    # -------------------------------------------------------------------------
    # 3. Pipeline Stage 1: Scout Training & Mining
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Scouts and Mining Hard Negatives...")

    # Train Scouts
    scout_a, scout_b = training_pipeline.train_scouts(X_train_full, y_train_full)

    assert scout_a.model is not None, "Scout A (LGBM) failed to train."
    assert scout_b.model is not None, "Scout B (XGB) failed to train."

    # Mine Hard Negatives
    # Note: With only 500 samples and random initialization, we might not find many hard negatives,
    # or we might find none if the model overfits perfectly. We handle this gracefully.
    hard_neg_indices = training_pipeline.mine_hard_negatives(
        X_train_full, y_train_full, scout_a, scout_b, load_cached_data=False
    )

    print(f"  Mined {len(hard_neg_indices)} hard negatives.")
    assert isinstance(
        hard_neg_indices, np.ndarray
    ), "Hard negative indices should be a numpy array."

    # -------------------------------------------------------------------------
    # 4. Pipeline Stage 2: Expert Dataset Construction
    # -------------------------------------------------------------------------
    print("\n[Step 4] Constructing Expert Dataset...")

    X_expert, y_expert = training_pipeline.construct_expert_dataset(
        X_train_full, y_train_full, hard_neg_indices
    )

    assert len(X_expert) > 0, "Expert dataset is empty."
    assert len(X_expert) == len(
        y_expert
    ), "Mismatch in X and y lengths for expert dataset."

    # -------------------------------------------------------------------------
    # 5. Pipeline Stage 3: Expert Training
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training Dual Ensemble Experts...")

    ensemble = training_pipeline.train_experts(X_expert, y_expert, X_val, y_val)

    assert ensemble.lgbm.model is not None, "Ensemble LGBM model not trained."
    assert ensemble.xgb.model is not None, "Ensemble XGB model not trained."

    # Test prediction capability
    sample_pred = ensemble.predict(X_val.iloc[:5])
    assert len(sample_pred) == 5, "Prediction output length mismatch."

    # -------------------------------------------------------------------------
    # 6. Pipeline Stage 4: Threshold Optimization
    # -------------------------------------------------------------------------
    print("\n[Step 6] Optimizing Threshold...")

    best_threshold = training_pipeline.optimize_threshold(ensemble, X_val, y_val)
    assert (
        0.0 < best_threshold < 1.0
    ), f"Threshold {best_threshold} is out of expected bounds."

    # -------------------------------------------------------------------------
    # 7. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Inference...")

    # Create a mini test metadata file to speed up inference demonstration
    # We'll take a slice of the train metadata and pretend it's test data
    # (In reality, test data has no labels, but the pipeline ignores them for features)
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test_metadata.csv")

    # Load original test metadata to get schema, take top 50 rows
    df_test_orig = pd.read_csv(config.TEST_METADATA_PATH, nrows=50)
    df_test_orig.to_csv(mini_test_path, index=False)

    # Override config path to point to our mini test set
    original_test_path = config.TEST_METADATA_PATH
    config.TEST_METADATA_PATH = mini_test_path

    # Also need to ensure sample_submission contains these IDs for the merge step in inference
    # We create a mini sample submission
    mini_sub_path = os.path.join(config.WORKING_DIR, "mini_sample_submission.csv")
    df_mini_sub = df_test_orig[["contact_id"]].copy()
    df_mini_sub["contact"] = 0
    df_mini_sub.to_csv(mini_sub_path, index=False)

    # Override sample submission path
    original_sub_path = config.SAMPLE_SUBMISSION_PATH
    config.SAMPLE_SUBMISSION_PATH = mini_sub_path

    try:
        # Run Inference
        # We pass load_cached_data=False so it regenerates features for our new mini file
        inference.run_inference(
            model=ensemble, threshold=best_threshold, load_cached_data=False
        )

        # Verify Submission File
        assert os.path.exists(
            config.SUBMISSION_FILE
        ), "Submission file was not created."

        df_sub = pd.read_csv(config.SUBMISSION_FILE)
        print(f"  Submission generated with {len(df_sub)} rows.")

        assert len(df_sub) == len(df_mini_sub), "Submission row count mismatch."
        assert (
            "contact_id" in df_sub.columns and "contact" in df_sub.columns
        ), "Submission schema incorrect."
        assert (
            df_sub["contact"].isin([0, 1]).all()
        ), "Submission contains non-binary values."

    finally:
        # Restore paths (good practice, though script ends here)
        config.TEST_METADATA_PATH = original_test_path
        config.SAMPLE_SUBMISSION_PATH = original_sub_path

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
