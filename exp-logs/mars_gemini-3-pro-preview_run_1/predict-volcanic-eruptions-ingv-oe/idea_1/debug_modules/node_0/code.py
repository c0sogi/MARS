import os
import pandas as pd
import numpy as np
import library.config as config
import library.dataset as dataset
import library.trainer as trainer
import library.inference as inference


def main():
    print("Initializing demonstration...")

    # ==========================================
    # 1. OPTIMIZATION: Create Mini Metadata Sets
    # ==========================================
    # To demonstrate the pipeline quickly, we use a small subset of the data.
    # We read the original metadata, sample it, and save to the working directory.

    # Define subset sizes
    N_TRAIN = 60
    N_VAL = 20
    N_TEST = 20

    print(f"Creating data subsets (Train={N_TRAIN}, Val={N_VAL}, Test={N_TEST})...")

    # Load original metadata
    orig_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample subsets (head is sufficient for demo)
    mini_train = orig_train.head(N_TRAIN).copy()
    mini_val = orig_val.head(N_VAL).copy()
    mini_test = orig_test.head(N_TEST).copy()

    # Save mini metadata
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Monkey-patch the config module to use our mini files
    # This ensures the library functions read our subsets instead of the full data
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    # Update cache directory to avoid messing with real caches
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "mini_cache")

    # ==========================================
    # 2. DATA LOADING & FEATURE ENGINEERING
    # ==========================================
    print("\n--- Step 2: Feature Engineering & Data Loading ---")

    # Load training and validation data
    # load_cached_data=False forces the feature extraction logic to run
    X_train, y_train, X_val, y_val = dataset.get_train_data(load_cached_data=False)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")

    # VALIDATION: Check if dimensions match our subset
    # Note: If some files were missing (handled in feature_engineering), count might be lower.
    # Based on provided metadata analysis, missing ratio is 0.0, so we expect exact matches.
    assert (
        len(X_train) == N_TRAIN
    ), f"Expected {N_TRAIN} training samples, got {len(X_train)}"
    assert len(X_val) == N_VAL, f"Expected {N_VAL} validation samples, got {len(X_val)}"
    assert len(y_train) == N_TRAIN

    # VALIDATION: Check feature columns
    # We expect stats (mean, std, min, max, etc.) for 10 sensors.
    # config.STATS_COLS has 14 stats. 10 sensors * 14 stats = 140 columns.
    expected_cols = config.NUM_SENSORS * len(config.STATS_COLS)
    assert (
        X_train.shape[1] == expected_cols
    ), f"Expected {expected_cols} features, got {X_train.shape[1]}"

    # ==========================================
    # 3. MODEL TRAINING
    # ==========================================
    print("\n--- Step 3: Model Training ---")

    # Define hyperparams for a very fast run
    fast_params = config.LGBM_PARAMS.copy()
    fast_params["num_leaves"] = 8  # Reduce complexity

    # Run Cross-Validation
    # We use 2 folds and 20 boosting rounds for speed
    models = trainer.run_cross_validation(
        X_train,
        y_train,
        params=fast_params,
        num_folds=2,
        num_boost_round=20,
        early_stopping_rounds=5,
        verbose_eval=10,
    )

    # VALIDATION: Ensure we got the correct number of models
    assert len(models) == 2, "Trainer should return a list of models equal to num_folds"

    # ==========================================
    # 4. INFERENCE & SUBMISSION
    # ==========================================
    print("\n--- Step 4: Inference & Submission ---")

    # Load test data
    X_test, test_ids = dataset.get_test_data(load_cached_data=False)

    print(f"X_test shape: {X_test.shape}")
    assert len(X_test) == N_TEST, f"Expected {N_TEST} test samples, got {len(X_test)}"

    # Generate predictions and save submission
    inference.predict_and_submit(models, X_test, test_ids)

    # ==========================================
    # 5. VERIFICATION OF SUBMISSION
    # ==========================================
    print("\n--- Step 5: Verification ---")

    submission_path = config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print("Submission file loaded successfully.")
    print(df_sub.head())

    # Check shape
    assert (
        len(df_sub) == N_TEST
    ), f"Submission should have {N_TEST} rows, found {len(df_sub)}"

    # Check columns
    expected_sub_cols = ["segment_id", "time_to_eruption"]
    assert (
        list(df_sub.columns) == expected_sub_cols
    ), f"Submission columns mismatch. Expected {expected_sub_cols}"

    # Check types
    assert pd.api.types.is_integer_dtype(
        df_sub["segment_id"]
    ), "segment_id should be integer"
    assert pd.api.types.is_numeric_dtype(
        df_sub["time_to_eruption"]
    ), "time_to_eruption should be numeric"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
