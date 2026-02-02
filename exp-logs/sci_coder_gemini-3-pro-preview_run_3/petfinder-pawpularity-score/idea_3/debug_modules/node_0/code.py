import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_extractor as feature_extractor
import library.ensemble_model as ensemble_model
import library.trainer as trainer


def run_demo():
    print("==================================================")
    print("       Pawpularity Prediction Library Demo        ")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed & Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use 50 samples per split

    # Reduce Cross-Validation folds
    Config.N_FOLDS = 2

    # Update Working and Submission Directories for isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Model Complexity for fast execution
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = None

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Testing Utilities
    # ---------------------------------------------------------
    print("\n[2] Testing Utilities...")

    # Test Seeding
    utils.seed_everything(42)

    # Test Logger
    log_path = os.path.join(Config.WORKING_DIR, "demo.log")
    logger = utils.setup_logger(log_path, name="demo_logger")
    logger.info("Logger test successful.")
    assert os.path.exists(log_path), "Log file was not created."

    # Test RMSE Calculation
    y_true = np.array([30.0, 50.0, 80.0])
    y_pred = np.array([32.0, 48.0, 85.0])
    rmse = utils.compute_rmse(y_true, y_pred)
    print(f"Computed RMSE (Dummy): {rmse:.4f}")
    assert rmse > 0, "RMSE calculation failed."

    # ---------------------------------------------------------
    # 3. Testing Data Loading
    # ---------------------------------------------------------
    print("\n[3] Testing Data Loader...")

    loaders = data_loader.get_dataloaders()
    assert "train" in loaders, "Train loader missing."
    assert "val" in loaders, "Val loader missing."

    # Fetch a single batch
    images, meta, targets, ids = next(iter(loaders["train"]))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Meta Shape:   {meta.shape}")
    print(f"Batch Targets Shape:{targets.shape}")

    # Verify Shapes
    # Image: (Batch, 3, 224, 224)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect Image Tensor shape."
    # Meta: (Batch, 12)
    assert meta.shape == (Config.BATCH_SIZE, 12), "Incorrect Metadata Tensor shape."
    # Target: (Batch,)
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect Target Tensor shape."

    # ---------------------------------------------------------
    # 4. Testing Feature Extraction
    # ---------------------------------------------------------
    print("\n[4] Testing Feature Extractor...")

    extractor = feature_extractor.FeatureExtractor()

    # Run extraction (computes and caches)
    # With DEBUG=True, this runs on the small subset
    print("Extracting features (this may take a moment for model initialization)...")
    data_dict = extractor.extract_and_cache(load_cached_data=False)

    assert "train" in data_dict
    assert "test" in data_dict

    train_feats = data_dict["train"]["features"]
    train_targets = data_dict["train"]["targets"]

    print(f"Extracted Train Features: {train_feats.shape}")

    # Verify dimensions
    # Rows should match DEBUG_SAMPLE_SIZE
    assert (
        train_feats.shape[0] == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {train_feats.shape[0]}"
    # Columns should be > 12 (Meta) + Backbones
    assert (
        train_feats.shape[1] > 12
    ), "Feature dimension too small, likely missing backbone features."

    # ---------------------------------------------------------
    # 5. Testing Ensemble Models
    # ---------------------------------------------------------
    print("\n[5] Testing Ensemble Models...")

    # --- Level 1 ---
    print("Testing Level 1 Predictors...")
    l1_model = ensemble_model.Level1Predictors()

    # Fit on the extracted features
    l1_model.fit(train_feats, train_targets)

    # Predict
    l1_preds = l1_model.predict(train_feats)
    print(f"Level 1 Predictions Shape: {l1_preds.shape}")

    # Should have 3 columns (SVR, LGBM, Ridge)
    assert l1_preds.shape == (len(train_targets), 3), "Level 1 output shape mismatch."

    # --- Level 2 ---
    print("Testing Level 2 Meta-Learner...")
    meta_learner = ensemble_model.MetaLearner()

    # Fit on Level 1 predictions
    meta_learner.fit(l1_preds, train_targets)

    # Predict
    final_preds = meta_learner.predict(l1_preds)
    print(f"Final Predictions Shape: {final_preds.shape}")

    assert final_preds.shape == (
        len(train_targets),
    ), "Meta-Learner output shape mismatch."

    # ---------------------------------------------------------
    # 6. Testing Full Trainer Pipeline
    # ---------------------------------------------------------
    print("\n[6] Testing Full Cross-Validation Trainer...")

    cv_trainer = trainer.CrossValidator()

    # Run the full pipeline
    # This will reuse the cached features from Step 4
    cv_trainer.run()

    # Verify Submission File
    print(f"Checking submission at: {Config.SUBMISSION_PATH}")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Verify Submission Content
    assert list(df_sub.columns) == ["Id", "Pawpularity"], "Submission columns mismatch."
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), "Submission row count mismatch (should match test debug size)."
    assert df_sub["Pawpularity"].isnull().sum() == 0, "Submission contains NaNs."

    print("\n==================================================")
    print("       Demo Completed Successfully!               ")
    print("==================================================")


if __name__ == "__main__":
    run_demo()
