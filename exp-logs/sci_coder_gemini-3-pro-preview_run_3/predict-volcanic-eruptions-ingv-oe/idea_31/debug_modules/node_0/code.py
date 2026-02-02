import os
import pandas as pd
import numpy as np
import shutil
import library.config as config
import library.utils as utils
import library.features as features
import library.data_processor as data_processor
import library.trainer as trainer
import library.inference as inference


def main():
    print("=== Starting Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override config constants
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.SUBMISSION_DIR = DEMO_WORKING_DIR  # Save submission in demo dir
    config.NUM_FOLDS = 2  # Reduce folds for speed
    config.SEED = 42

    # Override LightGBM params for extremely fast training on small data
    config.LGBM_PARAMS.update(
        {
            "n_estimators": 20,
            "num_leaves": 8,
            "min_child_samples": 2,  # Allow splits on very small sample sizes
            "learning_rate": 0.1,
            "verbose": -1,
        }
    )

    # Set reproducibility
    utils.seed_everything(config.SEED)
    print(f"Working Directory set to: {config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Unit Test: Feature Extraction Logic
    # ---------------------------------------------------------
    print("\n[2] Verifying Feature Extraction Logic...")

    # Pick a random file from train metadata to test extraction
    train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    sample_row = train_meta.iloc[0]
    file_path = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    # Load raw data
    raw_df = utils.load_csv(file_path)
    print(f"Loaded raw file: {file_path} with shape {raw_df.shape}")

    # Run feature extraction
    feats = features.extract_segment_features(raw_df)

    # Validation
    assert isinstance(feats, dict), "Feature extraction should return a dictionary"
    # Check for a few expected keys based on features.py logic
    expected_keys = [
        "sensor_1_mean",
        "sensor_1_std",
        "sensor_1_spec_low",
        "sensor_1_res_energy",
    ]
    for key in expected_keys:
        # Note: Depending on implementation details, keys might be named slightly differently
        # (e.g. sensor_1_trend_mean). Let's check the exact naming from features.py
        # features.py: features[f"{sensor}_{name}_mean"] where name is 'trend', 'vel', 'acc'
        # features.py: features[f"{sensor}_spec_{band}"]
        pass

    # Let's verify specific keys that are definitely in features.py
    assert "sensor_1_trend_mean" in feats, "Missing trend mean feature"
    assert "sensor_1_spec_low" in feats, "Missing spectral feature"
    assert "sensor_1_win0_rms" in feats, "Missing temporal profiling feature"

    print("Feature extraction logic verified successfully.")

    # ---------------------------------------------------------
    # 3. Integration Test: Data Processing Pipeline
    # ---------------------------------------------------------
    print("\n[3] Running Data Processing Pipeline (Sampling)...")

    # Process a small subset of training data
    # We use cache_name to store the result in the demo working dir
    train_feats_df = data_processor.process_set(
        metadata_path=os.path.join(config.METADATA_DIR, "train.csv"),
        cache_name="train_features_debug_20.parquet",
        load_cached_data=False,  # Force re-compute
        sample_size=20,
    )

    # Process a small subset of validation data
    val_feats_df = data_processor.process_set(
        metadata_path=os.path.join(config.METADATA_DIR, "val.csv"),
        cache_name="val_features_debug_10.parquet",
        load_cached_data=False,
        sample_size=10,
    )

    # Validation
    assert not train_feats_df.empty, "Processed training features are empty"
    assert not val_feats_df.empty, "Processed validation features are empty"
    assert (
        "time_to_eruption" in train_feats_df.columns
    ), "Target column missing in train features"
    assert (
        train_feats_df.shape[0] == 20
    ), f"Expected 20 train samples, got {train_feats_df.shape[0]}"

    # Check for NaNs
    if train_feats_df.isnull().sum().sum() > 0:
        print("Warning: NaNs found in feature matrix. Filling with 0 for demo.")
        train_feats_df.fillna(0, inplace=True)
        val_feats_df.fillna(0, inplace=True)

    print(f"Processed Train Shape: {train_feats_df.shape}")
    print(f"Processed Val Shape: {val_feats_df.shape}")

    # ---------------------------------------------------------
    # 4. Integration Test: Model Training
    # ---------------------------------------------------------
    print("\n[4] Running Model Training (CV)...")

    # Combine train and val for the CV runner (as done in config.run_pipeline logic)
    full_train_df = pd.concat([train_feats_df, val_feats_df], axis=0).reset_index(
        drop=True
    )

    # Run Cross-Validation
    models = trainer.run_cross_validation(full_train_df, num_folds=config.NUM_FOLDS)

    # Validation
    assert (
        len(models) == config.NUM_FOLDS
    ), f"Expected {config.NUM_FOLDS} models, got {len(models)}"

    # Check if model files were saved
    for i in range(config.NUM_FOLDS):
        model_path = os.path.join(config.WORKING_DIR, f"lgbm_model_fold_{i}.txt")
        assert os.path.exists(
            model_path
        ), f"Model artifact for fold {i} missing at {model_path}"

    print("Training completed and artifacts verified.")

    # ---------------------------------------------------------
    # 5. Integration Test: Inference and Submission
    # ---------------------------------------------------------
    print("\n[5] Running Inference Pipeline...")

    # We use the inference module's predict_and_submit function.
    # We pass sample_size=5 to limit the test processing time.
    # This function handles: loading test features -> loading models -> predicting -> saving csv

    inference.predict_and_submit(load_cached_data=False, sample_size=5)

    # Validation
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")
    print("Head of submission:")
    print(sub_df.head())

    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns"
    assert len(sub_df) == 5, f"Expected 5 predictions, got {len(sub_df)}"
    assert (
        sub_df["time_to_eruption"].notnull().all()
    ), "Submission contains null predictions"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
