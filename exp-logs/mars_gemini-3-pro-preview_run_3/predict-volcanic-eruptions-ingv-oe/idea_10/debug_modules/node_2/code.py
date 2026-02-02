import os
import sys
import pandas as pd
import numpy as np
import joblib
import shutil

# Import the provided library modules
# We assume the script is running from the root directory where 'library' package exists
import library.config as config
import library.feature_engineering as fe
import library.data_processor as dp
import library.models as models
import library.trainer as trainer

# Set seeds for reproducibility in the demo script
np.random.seed(42)


def demo_feature_engineering():
    print("\n=== Demo 1: Feature Engineering (Single Segment) ===")

    # Load metadata to get a valid file path
    train_meta_path = os.path.join(config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    train_meta = pd.read_csv(train_meta_path)

    # Pick the first segment
    sample_row = train_meta.iloc[0]
    segment_id = sample_row["segment_id"]
    file_path = sample_row["file_path"]

    print(f"Processing segment: {segment_id} from {file_path}")

    # Execute feature extraction
    features = fe.process_segment(file_path, segment_id)

    # Validation
    assert isinstance(features, dict), "Output should be a dictionary"
    assert features["segment_id"] == segment_id, "Segment ID mismatch"

    # Check for specific feature groups mentioned in the library
    keys = features.keys()
    has_raw = any("raw_min" in k for k in keys)
    has_smooth = any("smooth_mean" in k for k in keys)
    has_spec = any("spec_mean" in k for k in keys)
    has_win = any("win_0_rms" in k for k in keys)

    assert has_raw, "Missing Raw Extrema features"
    assert has_smooth, "Missing Kinematic features"
    assert has_spec, "Missing Spectral features"
    assert has_win, "Missing Temporal Window features"

    print(f"Successfully extracted {len(features)} features.")
    print("Feature Engineering logic verified.")


def demo_data_processor():
    print("\n=== Demo 2: Data Processor (Batch Generation) ===")

    debug_size = 10
    dataset_type = "train"

    # Clean up any existing cache for this test to ensure fresh generation
    cache_path = os.path.join(
        config.WORKING_DIR, f"{dataset_type}_features_debug_{debug_size}.parquet"
    )
    if os.path.exists(cache_path):
        os.remove(cache_path)

    print(
        f"Generating feature matrix for {dataset_type} with debug_size={debug_size}..."
    )

    # Execute data processor
    df = dp.generate_feature_matrix(
        dataset_type, load_cached=False, debug_size=debug_size
    )

    # Validation
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert len(df) == debug_size, f"Expected {debug_size} rows, got {len(df)}"
    assert "segment_id" in df.columns, "DataFrame missing segment_id column"

    # Check if cache file was created
    assert os.path.exists(cache_path), f"Cache file not created at {cache_path}"

    print(f"Generated DataFrame shape: {df.shape}")
    print("Data Processor logic verified.")


def monkey_patch_trainer_for_speed():
    """
    To ensure the pipeline runs quickly for demonstration purposes, we monkey-patch
    the 'get_base_models' function and 'N_FOLDS' variable inside the library.trainer module.
    This replaces the computationally expensive models (2000 estimators) with lightweight versions.
    """
    print("\n=== Configuring Pipeline for Speed (Monkey Patching) ===")

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    # Define a fast model generator
    def fast_get_base_models(random_seed=42, n_jobs=1):
        print(">> Using FAST mocked models for demonstration")
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=10,  # Reduced from 2000
            learning_rate=0.05,
            num_leaves=10,
            random_state=random_seed,
            verbosity=-1,
            n_jobs=n_jobs,
        )

        xgb_model = xgb.XGBRegressor(
            n_estimators=10,  # Reduced from 2000
            learning_rate=0.05,
            max_depth=3,
            random_state=random_seed,
            n_jobs=n_jobs,
            tree_method="hist",
            early_stopping_rounds=5,
        )

        cat_model = CatBoostRegressor(
            iterations=10,  # Reduced from 2000
            learning_rate=0.05,
            depth=3,
            random_seed=random_seed,
            verbose=0,
            allow_writing_files=False,
            thread_count=n_jobs,
        )

        return {"lgbm": lgbm_model, "xgb": xgb_model, "cat": cat_model}

    # Apply patches
    trainer.get_base_models = fast_get_base_models
    trainer.N_FOLDS = 2  # Reduced from 5

    print("Trainer module patched: N_FOLDS=2, n_estimators=10.")


def demo_full_pipeline():
    print("\n=== Demo 3: Full Training Pipeline Execution ===")

    # We use a slightly larger debug size to ensure we have enough data for a 2-fold split
    # 20 samples -> 10 train / 10 val per fold
    debug_size = 20

    # Apply speed optimizations
    monkey_patch_trainer_for_speed()

    # Clean up submission path to verify generation later
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    print(f"Running pipeline with debug_size={debug_size}...")

    # Execute Pipeline
    # load_cached=True allows reusing the features if they exist (though we might have just deleted them in Demo 2)
    # The trainer handles feature generation for train/val/test internally.
    trainer.run_training_pipeline(debug_size=debug_size, load_cached=True)

    # Validation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {submission_df.shape}")

    assert (
        len(submission_df) == debug_size
    ), f"Submission should have {debug_size} rows (matching test debug size)"
    assert "segment_id" in submission_df.columns, "Submission missing segment_id"
    assert (
        "time_to_eruption" in submission_df.columns
    ), "Submission missing time_to_eruption"

    # Basic sanity check on predictions
    preds = submission_df["time_to_eruption"]
    assert not preds.isnull().any(), "Submission contains null predictions"

    print("Full Pipeline executed successfully.")


if __name__ == "__main__":
    try:
        # 1. Verify Feature Engineering
        demo_feature_engineering()

        # 2. Verify Data Processor
        demo_data_processor()

        # 3. Verify Full Pipeline (with speed optimizations)
        demo_full_pipeline()

        print("\nAll demonstrations passed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
