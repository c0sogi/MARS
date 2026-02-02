import os
import pandas as pd
import numpy as np
import library.config as config
import library.features as features
import library.data_loader as data_loader
import library.model_trainer as model_trainer


def main():
    print("=== Starting Volcano Eruption Prediction Demonstration ===")

    # Ensure reproducibility
    np.random.seed(config.SEED)

    # Define paths for temporary subset metadata to ensure speed
    mini_train_meta_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_meta_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_meta_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    # 1. Create Data Subsets
    # We load the full metadata and sample a small number of rows to avoid
    # processing thousands of sensor files during this demonstration.
    print("\n[1] Creating metadata subsets for fast processing...")

    full_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    full_test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # Take top 50 for train, 20 for val, 20 for test
    full_train_meta.head(50).to_csv(mini_train_meta_path, index=False)
    full_val_meta.head(20).to_csv(mini_val_meta_path, index=False)
    full_test_meta.head(20).to_csv(mini_test_meta_path, index=False)

    print(f"Subsets created in {config.WORKING_DIR}")

    # 2. Feature Engineering
    # We use features.process_dataset directly with our mini metadata files.
    # In a full run, data_loader.build_dataset would be used with the default paths.
    print("\n[2] Processing features from sensor data...")

    # Process Train
    df_train = features.process_dataset(
        metadata_path=mini_train_meta_path,
        load_cached_data=False,  # Force computation to demonstrate logic
        save_name="mini_train_features",
    )

    # Process Val
    df_val = features.process_dataset(
        metadata_path=mini_val_meta_path,
        load_cached_data=False,
        save_name="mini_val_features",
    )

    # Process Test
    df_test = features.process_dataset(
        metadata_path=mini_test_meta_path,
        load_cached_data=False,
        save_name="mini_test_features",
    )

    # Validation of Feature Extraction
    assert not df_train.empty, "Training dataframe should not be empty."
    assert "segment_id" in df_train.columns, "segment_id column missing from features."
    assert (
        "time_to_eruption" in df_train.columns
    ), "Target column missing from training features."
    # Check that we have sensor features (e.g., sensor_1_mean)
    assert any(
        c.startswith("sensor_") for c in df_train.columns
    ), "No sensor features detected."

    print(f"Train Features Shape: {df_train.shape}")
    print(f"Val Features Shape: {df_val.shape}")
    print(f"Test Features Shape: {df_test.shape}")

    # 3. Data Preparation
    # Split into X (features) and y (target) using the data_loader utility
    print("\n[3] Preparing data for training...")

    X_train, y_train = data_loader.prepare_features_target(df_train)
    X_val, y_val = data_loader.prepare_features_target(df_val)

    # Verify separation
    assert "segment_id" not in X_train.columns, "segment_id should be removed from X."
    assert "time_to_eruption" not in X_train.columns, "Target should be removed from X."
    assert len(X_train) == len(y_train), "Mismatch between features and target length."
    assert len(X_train) == 50, "Expected 50 training samples based on subset."

    # 4. Model Training
    # Train the regressor. We limit n_estimators to 10 for speed.
    print("\n[4] Training LightGBM Regressor...")

    model, mae = model_trainer.train_regressor(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=10,  # Reduced for demo speed
        random_state=config.SEED,
    )

    print(f"Training completed. Validation MAE: {mae:.4f}")

    # Verify model outputs
    assert model is not None, "Model object is None."
    assert isinstance(mae, float), "MAE should be a float."
    assert mae >= 0, "MAE cannot be negative."

    # 5. Submission Generation
    # Generate predictions for the test set
    print("\n[5] Generating submission file...")

    submission_output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    model_trainer.generate_submission(
        model, df_test, output_path=submission_output_path
    )

    # Verify Submission File
    assert os.path.exists(submission_output_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_output_path)
    print(f"Submission file loaded. Shape: {sub_df.shape}")

    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns."
    assert len(sub_df) == 20, "Submission should have 20 rows based on test subset."
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
