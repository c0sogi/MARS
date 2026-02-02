import os
import shutil
import warnings
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config, utils, data_loader, feature_extractor, svr_model


def main():
    print("=== Pet Pawpularity Prediction Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Ensure reproducibility
    utils.seed_everything(seed=42)

    # Override Config for Speed and Isolation
    print("\n[Setup] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use only 100 samples per dataset
    config.DEBUG = True

    # Create a separate working directory for this demo to avoid conflicts
    config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Update paths in config to point to the new working directory
    # Note: Modifying the config module object affects all other modules importing it
    config.TRAIN_FEATURES_PATH = os.path.join(config.WORKING_DIR, "train_features.npy")
    config.TRAIN_TARGETS_PATH = os.path.join(config.WORKING_DIR, "train_targets.npy")
    config.VAL_FEATURES_PATH = os.path.join(config.WORKING_DIR, "val_features.npy")
    config.VAL_TARGETS_PATH = os.path.join(config.WORKING_DIR, "val_targets.npy")
    config.TEST_FEATURES_PATH = os.path.join(config.WORKING_DIR, "test_features.npy")
    config.TEST_IDS_PATH = os.path.join(config.WORKING_DIR, "test_ids.npy")
    config.SVR_MODEL_PATH = os.path.join(config.WORKING_DIR, "svr_model.joblib")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Optimize Grid Search for speed (Single iteration)
    config.SVR_GRID = {"C": [1.0], "epsilon": [0.1]}
    config.N_FOLDS = 2  # Reduce folds for speed

    print(f"Working Directory: {config.WORKING_DIR}")
    print("Debug Mode: Enabled (100 samples per subset)")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Data] Initializing DataLoaders...")
    # Use num_workers=0 to avoid multiprocessing overhead in this short script
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=16, num_workers=0, debug=config.DEBUG
    )

    # Verification: Check batch dimensions
    sample_imgs, sample_meta, sample_targets = next(iter(train_loader))
    print(
        f"Sample Batch - Image: {sample_imgs.shape}, Meta: {sample_meta.shape}, Target: {sample_targets.shape}"
    )

    # Assertions to ensure data loader logic is correct
    assert sample_imgs.shape == (16, 3, 224, 224), "Incorrect Image Tensor shape"
    assert sample_meta.shape == (
        16,
        12,
    ), "Incorrect Metadata Tensor shape (Expected 12 binary features)"
    assert sample_targets.shape == (16,), "Incorrect Target Tensor shape"

    # ---------------------------------------------------------
    # 3. Feature Extraction
    # ---------------------------------------------------------
    print(
        "\n[Feature Extraction] Extracting features from backbones (Swin + EffNet)..."
    )

    # Extract Train Features
    # We force load_cached_data=False to demonstrate the extraction process
    print("Processing Train Set...")
    train_features, train_targets = feature_extractor.get_features(
        train_loader, mode="train", load_cached_data=False
    )

    # Extract Val Features
    print("Processing Validation Set...")
    val_features, val_targets = feature_extractor.get_features(
        val_loader, mode="val", load_cached_data=False
    )

    print(f"Train Features Shape: {train_features.shape}")
    print(f"Val Features Shape: {val_features.shape}")

    # Assertions
    assert (
        train_features.shape[0] == 100
    ), "Train feature count mismatch (expected 100 in debug mode)"
    assert (
        val_features.shape[0] == 100
    ), "Val feature count mismatch (expected 100 in debug mode)"
    # Check for NaN values which would indicate model or processing failure
    assert not np.isnan(train_features).any(), "NaN values found in train features"

    # ---------------------------------------------------------
    # 4. Model Training (SVR)
    # ---------------------------------------------------------
    print("\n[Training] Fitting SVR Model with GridSearchCV...")
    regressor = svr_model.PetPawpularityRegressor()

    # Fit the model
    regressor.fit(train_features, train_targets)

    # Verify model persistence
    assert os.path.exists(config.SVR_MODEL_PATH), "Model file was not saved to disk"

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    print("\n[Evaluation] Predicting on Validation Set...")
    val_preds = regressor.predict(val_features)

    # Calculate Metric
    rmse = utils.calculate_rmse(val_targets, val_preds)
    print(f"Validation RMSE: {rmse:.4f}")

    # Assertions
    assert len(val_preds) == 100, "Prediction count mismatch"
    assert np.all(
        (val_preds >= 1.0) & (val_preds <= 100.0)
    ), "Predictions out of valid range [1, 100]"

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[Inference] Generating Submission for Test Set...")

    # Extract Test Features
    test_features, test_ids = feature_extractor.get_features(
        test_loader, mode="test", load_cached_data=False
    )

    # Predict
    test_preds = regressor.predict(test_features)

    # Generate CSV
    svr_model.generate_submission(
        test_ids, test_preds, output_path=config.SUBMISSION_PATH
    )

    # Verify Submission
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission File Saved. Shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())

        # Assertions
        assert df_sub.shape == (100, 2), "Submission shape mismatch (expected 100 rows)"
        assert list(df_sub.columns) == [
            "Id",
            "Pawpularity",
        ], "Submission columns mismatch"
        assert not df_sub.isnull().values.any(), "Submission contains null values"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
