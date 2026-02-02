import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import library components
import library.config as config
from library.utils import seed_everything
from library.feature_engineering import process_tabular_data
from library.trainer_lgbm import train_lgbm_model, predict_lgbm
from library.data_processing import get_spectrogram_loaders
from library.model_cnn import SeismicCNN
from library.trainer_cnn import run_cnn_training, predict_cnn

# =============================================================================
# 0. CONFIGURATION OVERRIDES FOR SPEED
# =============================================================================
# We modify the global configuration to run a fast demonstration
print("--- Configuring environment for fast demonstration ---")

# Reduce CNN training duration
config.EPOCHS = 1
config.DEBUG_SAMPLE_SIZE = 20  # Only use 20 samples for debug mode
config.BATCH_SIZE = 4

# Reduce LightGBM complexity
config.LGBM_PARAMS["n_estimators"] = 10
config.LGBM_PARAMS["verbosity"] = -1

# Set paths for mini-metadata (used for Tabular demo)
MINI_TRAIN_META = os.path.join(config.WORKING_DIR, "mini_train.csv")
MINI_VAL_META = os.path.join(config.WORKING_DIR, "mini_val.csv")
MINI_TEST_META = os.path.join(config.WORKING_DIR, "mini_test.csv")

seed_everything(config.SEED)

if __name__ == "__main__":

    # =========================================================================
    # 1. PREPARE MINI METADATA (For Tabular Stream)
    # =========================================================================
    print("\n[Step 1] Preparing mini metadata files...")

    # Read original metadata
    full_train = pd.read_csv(config.TRAIN_META_PATH)
    full_val = pd.read_csv(config.VAL_META_PATH)
    full_test = pd.read_csv(config.TEST_META_PATH)

    # Sample a small subset
    mini_train = full_train.head(config.DEBUG_SAMPLE_SIZE).copy()
    mini_val = full_val.head(config.DEBUG_SAMPLE_SIZE).copy()
    mini_test = full_test.head(config.DEBUG_SAMPLE_SIZE).copy()

    # Save to working directory
    mini_train.to_csv(MINI_TRAIN_META, index=False)
    mini_val.to_csv(MINI_VAL_META, index=False)
    mini_test.to_csv(MINI_TEST_META, index=False)

    assert os.path.exists(MINI_TRAIN_META), "Mini train metadata not created"
    print(f"Created mini metadata with {len(mini_train)} samples each.")

    # =========================================================================
    # 2. STREAM A: TABULAR FEATURE ENGINEERING & LIGHTGBM
    # =========================================================================
    print("\n[Step 2] Running Stream A (Tabular + LightGBM)...")

    # 2.1 Generate Features
    # We use the mini metadata files created above
    print("Generating tabular features...")
    train_feat = process_tabular_data(
        MINI_TRAIN_META, "mini_train_features.parquet", load_cached_data=False
    )
    val_feat = process_tabular_data(
        MINI_VAL_META, "mini_val_features.parquet", load_cached_data=False
    )
    test_feat = process_tabular_data(
        MINI_TEST_META, "mini_test_features.parquet", load_cached_data=False
    )

    # Verify Feature Generation
    assert not train_feat.empty, "Train feature DataFrame is empty"
    assert "sensor_1_mean" in train_feat.columns, "Statistical features missing"
    assert (
        "time_to_eruption" in train_feat.columns
    ), "Target variable missing from train features"
    print(f"Tabular features generated. Shape: {train_feat.shape}")

    # 2.2 Train LightGBM
    print("Training LightGBM model...")
    lgbm_model, val_preds = train_lgbm_model(train_feat, val_feat)

    # Verify Training
    assert lgbm_model is not None, "LightGBM model is None"
    assert len(val_preds) == len(val_feat), "Validation prediction length mismatch"

    # 2.3 Predict on Test
    print("Predicting on test set with LightGBM...")
    lgbm_test_preds = predict_lgbm(lgbm_model, test_feat)

    assert len(lgbm_test_preds) == len(test_feat), "Test prediction length mismatch"
    print("Stream A completed successfully.")

    # =========================================================================
    # 3. STREAM B: SPECTROGRAMS & CNN
    # =========================================================================
    print("\n[Step 3] Running Stream B (Spectrograms + CNN)...")

    # 3.1 Data Loading Check
    # We use debug=True to leverage the DEBUG_SAMPLE_SIZE we set in config
    print("Initializing DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = get_spectrogram_loaders(
        batch_size=config.BATCH_SIZE, debug=True
    )

    # Verify Data Loading
    sample_batch = next(iter(train_loader))
    imgs, targets, ids = sample_batch
    # Expected shape: (Batch, Channels, Height, Width) -> (4, 10, 224, 224)
    print(f"Batch Shape: {imgs.shape}")
    assert imgs.shape == (
        config.BATCH_SIZE,
        10,
        224,
        224,
    ), f"Incorrect input shape: {imgs.shape}"
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Incorrect target shape: {targets.shape}"

    # 3.2 Model Instantiation Check
    print("Instantiating SeismicCNN...")
    cnn_model = SeismicCNN()
    # Test forward pass on CPU
    with torch.no_grad():
        out = cnn_model(imgs)
    assert out.shape == (config.BATCH_SIZE, 1), f"Incorrect output shape: {out.shape}"
    print("Model forward pass verified.")

    # 3.3 Train CNN
    # run_cnn_training handles the training loop, validation, and saving best model
    print("Starting CNN Training Loop...")
    trained_cnn, best_mae = run_cnn_training(debug=True)

    assert isinstance(best_mae, float), "Best MAE is not a float"
    print(f"CNN Training finished. Best Val MAE: {best_mae:.4f}")

    # 3.4 Predict on Test
    # We pass the test_loader we created earlier to ensure we predict on the mini set
    print("Predicting on test set with CNN...")
    cnn_results_df = predict_cnn(trained_cnn, test_loader=test_loader)

    assert "time_to_eruption" in cnn_results_df.columns, "Missing prediction column"
    assert (
        len(cnn_results_df) == config.DEBUG_SAMPLE_SIZE
    ), "CNN prediction count mismatch"

    # Sort by segment_id to ensure alignment with tabular predictions
    cnn_results_df = cnn_results_df.sort_values("segment_id").reset_index(drop=True)
    cnn_test_preds = cnn_results_df["time_to_eruption"].values
    print("Stream B completed successfully.")

    # =========================================================================
    # 4. ENSEMBLE & SUBMISSION
    # =========================================================================
    print("\n[Step 4] Ensembling and Generating Submission...")

    # Ensure alignment
    # Tabular test_feat also needs to be sorted by segment_id
    test_feat_sorted = test_feat.sort_values("segment_id").reset_index(drop=True)

    # Verify IDs match
    assert np.all(
        test_feat_sorted["segment_id"].values == cnn_results_df["segment_id"].values
    ), "Segment ID mismatch between streams"

    # Re-predict LGBM on sorted data to be safe
    lgbm_test_preds_sorted = predict_lgbm(lgbm_model, test_feat_sorted)

    # Weighted Average
    w_lgbm = config.ENSEMBLE_WEIGHT
    w_cnn = 1.0 - config.ENSEMBLE_WEIGHT

    final_preds = (w_lgbm * lgbm_test_preds_sorted) + (w_cnn * cnn_test_preds)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": test_feat_sorted["segment_id"], "time_to_eruption": final_preds}
    )

    # Save
    sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print("Head of submission:")
    print(submission_df.head())

    print("\nAll demonstration steps completed successfully.")
