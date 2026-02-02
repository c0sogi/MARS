import os
import pandas as pd
import numpy as np
import shutil
import joblib
from library.config import Config
from library.utils import seed_everything
from library.features import extract_sensor_features
from library.data_loader import load_sensor_file
from library.training_pipeline import run_training, generate_submission_file


def main():
    # 1. Setup and Configuration
    print("--- Setting up environment ---")
    seed_everything(42)

    # Define a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config to point to this directory
    Config.WORKING_DIR = demo_dir

    # Override Hyperparameters for Speed (Fast Mode)
    print("--- Overriding configuration for fast execution ---")
    Config.N_FOLDS = 2
    Config.EARLY_STOPPING = 10

    # Reduce estimators for all models
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.CAT_PARAMS["iterations"] = 10

    # 2. Unit Verification: Feature Extraction
    print("\n--- Verifying Feature Extraction Logic ---")
    # Load train metadata to find a valid file
    train_meta = pd.read_csv(Config.TRAIN_META)
    sample_row = train_meta.iloc[0]
    sample_file_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    # Test loading
    df_sample = load_sensor_file(sample_file_path)
    assert not df_sample.empty, "Failed to load sample sensor file."

    # Test feature extraction on one sensor
    sensor_name = "sensor_1"
    if sensor_name in df_sample.columns:
        feats = extract_sensor_features(df_sample[sensor_name].values, sensor_name)

        # Check for expected keys (Stream A and Stream B features)
        expected_keys = [
            f"{sensor_name}_mean",
            f"{sensor_name}_raw_min",
            f"{sensor_name}_spec_low",
            f"{sensor_name}_vel_std",
        ]
        for k in expected_keys:
            assert k in feats, f"Missing feature key: {k}"
            assert not np.isnan(feats[k]), f"Feature {k} is NaN"

        print(
            f"Feature extraction verified. Extracted {len(feats)} features for {sensor_name}."
        )

    # 3. Pipeline Execution: Training
    print("\n--- Running Training Pipeline (Debug Mode) ---")
    # run_training with debug=True processes only the first 50 rows of metadata
    s1_models, s2_models, s3_model, s2_features = run_training(
        load_cached_data=False, debug=True
    )

    # Verify artifacts exist
    assert os.path.exists(
        os.path.join(demo_dir, "stage1_models.pkl")
    ), "Stage 1 models not saved."
    assert os.path.exists(
        os.path.join(demo_dir, "stage2_models.pkl")
    ), "Stage 2 models not saved."
    assert os.path.exists(
        os.path.join(demo_dir, "stage3_model.pkl")
    ), "Stage 3 model not saved."
    print("Training completed and models saved.")

    # 4. Pipeline Execution: Inference (Submission)
    print("\n--- Preparing Mini Test Set for Inference ---")
    # Create a mini test metadata file to speed up submission generation
    test_meta_orig = pd.read_csv(Config.TEST_META)
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    mini_test_df = test_meta_orig.head(10).copy()  # Process only 10 test files
    mini_test_df.to_csv(mini_test_path, index=False)

    # Override Config to point to mini test set
    Config.TEST_META = mini_test_path

    print("--- Generating Submission ---")
    generate_submission_file(load_cached_data=False)

    # 5. Final Validation
    print("\n--- Validating Submission File ---")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    # Checks
    assert len(sub_df) == 10, f"Expected 10 predictions, got {len(sub_df)}"
    assert "segment_id" in sub_df.columns and "time_to_eruption" in sub_df.columns
    assert not sub_df["time_to_eruption"].isnull().any(), "Submission contains NaNs"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
