import os
import sys
import pandas as pd
import numpy as np
import shutil

# Import provided library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.data_processor as data_processor
import library.model_trainer as model_trainer


def main():
    print("Starting demonstration and verification script...")

    # ==========================================
    # 1. Configuration Override for Speed/Demo
    # ==========================================
    print("\n--- Step 1: Configuring Environment for Demo ---")

    # Enable Debug mode to use a small subset of data
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50  # Small sample for fast execution

    # Reduce Cross-Validation Folds
    config.N_FOLDS = 2

    # Reduce Model Complexity for Speed
    # LightGBM
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["verbose"] = -1

    # XGBoost
    config.XGB_PARAMS["n_estimators"] = 10

    # Reduce Early Stopping Rounds
    config.EARLY_STOPPING_ROUNDS = 5

    print(f"DEBUG Mode: {config.DEBUG}")
    print(f"Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"N_FOLDS: {config.N_FOLDS}")
    print("Model estimators reduced to 10 for speed.")

    # Set Seeds
    utils.seed_everything(config.SEED)

    # ==========================================
    # 2. Metadata Loading & Verification
    # ==========================================
    print("\n--- Step 2: Verifying Metadata Loading ---")

    train_df, val_df, test_df = utils.load_metadata()

    print(f"Original Train shape: {train_df.shape}")
    print(f"Original Val shape: {val_df.shape}")
    print(f"Original Test shape: {test_df.shape}")

    assert not train_df.empty, "Train metadata is empty"
    assert not val_df.empty, "Val metadata is empty"
    assert not test_df.empty, "Test metadata is empty"
    assert "segment_id" in train_df.columns
    assert "time_to_eruption" in train_df.columns
    assert "file_path" in train_df.columns

    # ==========================================
    # 3. Feature Extraction Verification
    # ==========================================
    print("\n--- Step 3: Verifying Feature Extraction Logic ---")

    # Pick a sample file from training metadata
    sample_row = train_df.iloc[0]
    segment_id = int(sample_row["segment_id"])
    rel_path = sample_row["file_path"]
    full_path = os.path.join(config.INPUT_DIR, rel_path)

    print(f"Extracting features for segment {segment_id} from {full_path}...")

    # Extract features
    single_feats = features.extract_features_for_segment(full_path, segment_id)

    # Validations
    assert isinstance(
        single_feats, dict
    ), "Feature extraction should return a dictionary"
    assert single_feats["segment_id"] == segment_id, "Segment ID mismatch in features"

    # Check for specific feature groups
    keys = single_feats.keys()
    has_kinematic = any("vel_mean" in k for k in keys)
    has_spectral = any("spec_centroid" in k for k in keys)
    has_spatial = any("corr_" in k for k in keys)

    assert has_kinematic, "Kinematic features missing"
    assert has_spectral, "Spectral features missing"
    assert has_spatial, "Spatial features missing"

    # Check for NaNs
    values = list(single_feats.values())
    assert not np.isnan(values).any(), "Features contain NaN values"

    print("Feature extraction verified successfully.")

    # ==========================================
    # 4. Data Processing Pipeline Verification
    # ==========================================
    print("\n--- Step 4: Verifying Data Processor (Train/Val Generation) ---")

    # This will use the DEBUG_SAMPLE_SIZE and generate features
    # We set load_cached_data=False to ensure the generation logic runs
    X_train, y_train, X_val, y_val = data_processor.get_train_val_datasets(
        load_cached_data=False
    )

    print(f"Generated X_train shape: {X_train.shape}")
    print(f"Generated y_train shape: {y_train.shape}")

    # In DEBUG mode, we sample DEBUG_SAMPLE_SIZE for *both* train and val independently in the code logic
    # (See library/data_processor.py: get_train_val_datasets implementation)
    # The code samples the metadata DF.
    expected_size = min(len(train_df), config.DEBUG_SAMPLE_SIZE)
    assert (
        len(X_train) == expected_size
    ), f"Expected {expected_size} training samples, got {len(X_train)}"
    assert len(y_train) == expected_size
    assert not X_train.isnull().values.any(), "X_train contains NaNs"

    print("Data processing pipeline verified.")

    # ==========================================
    # 5. Model Training & Ensemble Verification
    # ==========================================
    print("\n--- Step 5: Verifying Model Training and Ensemble ---")

    # Clean up previous submission if exists
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        os.remove(submission_path)

    # Run the full ensemble pipeline
    # This will:
    # 1. Load the datasets (cached from Step 4)
    # 2. Generate test features (small subset due to DEBUG)
    # 3. Run Stratified K-Fold (2 folds)
    # 4. Train LGBM and XGB
    # 5. Save submission
    model_trainer.run_stratified_ensemble(load_cached_data=True)

    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    # Expected test size in debug mode
    expected_test_size = min(len(test_df), config.DEBUG_SAMPLE_SIZE)
    assert (
        len(sub_df) == expected_test_size
    ), f"Submission row count mismatch. Expected {expected_test_size}, got {len(sub_df)}"
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns incorrect"
    assert sub_df["time_to_eruption"].dtype in [
        float,
        np.float32,
        np.float64,
    ], "Prediction column is not float"

    print("\nModel training and submission generation verified successfully.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
