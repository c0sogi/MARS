import os
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import the provided library modules
from library import config, utils, network, data_loader, train, predict


def run_demo():
    print("--- Starting End-to-End Demo ---")

    # =========================================================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # =========================================================================
    # Create a separate directory for demo outputs to ensure isolation and speed
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Configuring demo environment in: {DEMO_DIR}")

    # Override configuration constants to use the demo directory
    # Note: We must update derived paths as well since they were initialized on import
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = DEMO_DIR

    # Update file paths
    config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "demo_submission.csv")
    config.TRAIN_PATCHES_PATH = os.path.join(DEMO_DIR, "train_patches.npy")
    config.TRAIN_TARGETS_PATH = os.path.join(DEMO_DIR, "train_targets.npy")
    config.VAL_PATCHES_PATH = os.path.join(DEMO_DIR, "val_patches.npy")
    config.VAL_TARGETS_PATH = os.path.join(DEMO_DIR, "val_targets.npy")

    # Set random seed for reproducibility
    utils.seed_everything(42)

    # =========================================================================
    # 2. VERIFY UTILITIES
    # =========================================================================
    print("\n[1/5] Verifying Utility Functions...")

    # Test Normalization
    dummy_img_uint8 = np.array([[0, 128, 255]], dtype=np.uint8)
    norm_img = utils.normalize_image(dummy_img_uint8)
    assert norm_img.dtype == np.float32, "Normalized image should be float32"
    assert norm_img.min() >= 0.0 and norm_img.max() <= 1.0, "Image not in [0, 1]"

    denorm_img = utils.denormalize_image(norm_img)
    assert denorm_img.dtype == np.uint8, "Denormalized image should be uint8"
    # Allow small rounding differences, but exact match expected for 0 and 255
    assert denorm_img[0, 0] == 0 and denorm_img[0, 2] == 255

    # Test RMSE Calculation
    preds = np.array([0.2, 0.8])
    targets = np.array([0.2, 0.8])
    rmse_perfect = utils.calculate_rmse(preds, targets)
    assert rmse_perfect == 0.0, "RMSE should be 0 for perfect predictions"

    preds_off = np.array([0.0, 1.0])
    targets_off = np.array([1.0, 0.0])
    # MSE = 1, RMSE = 1
    rmse_bad = utils.calculate_rmse(preds_off, targets_off)
    assert np.isclose(rmse_bad, 1.0), "RMSE calculation incorrect"

    print(" - Utils verified successfully.")

    # =========================================================================
    # 3. VERIFY NETWORK ARCHITECTURE
    # =========================================================================
    print("\n[2/5] Verifying Network Architecture...")

    device = torch.device("cpu")  # Use CPU for quick structural check
    # Instantiate a smaller version of the model for the demo
    model = network.SE_ZI_ResDnCNN(
        in_channels=1,
        out_channels=1,
        num_features=16,
        num_blocks=2,
        zero_init_residual=True,
    ).to(device)

    # Check parameter count
    params = utils.count_parameters(model)
    print(f" - Model instantiated with {params} parameters.")
    assert params > 0, "Model has no parameters"

    # Check Forward Pass
    # Input shape: (Batch, Channel, Height, Width)
    dummy_input = torch.randn(2, 1, 64, 64).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == dummy_input.shape, f"Output shape mismatch: {output.shape}"
    print(" - Forward pass shape verified.")

    # =========================================================================
    # 4. VERIFY DATA PIPELINE
    # =========================================================================
    print("\n[3/5] Verifying Data Pipeline...")

    # Process a tiny subset of data (2 images)
    # This will generate and save cache files in DEMO_DIR
    train_patches, train_targets, val_patches, val_targets = data_loader.prepare_data(
        load_cached_data=False, max_samples=2
    )

    assert len(train_patches) > 0, "No patches extracted"
    assert train_patches.shape == train_targets.shape, "Patch/Target shape mismatch"
    print(f" - Extracted {len(train_patches)} patches from subset.")

    # Verify Dataset Class
    ds = data_loader.DenoisingDataset(train_patches, train_targets, augment=True)
    sample_x, sample_y = ds[0]
    # Expecting (1, PatchSize, PatchSize) tensors
    assert sample_x.ndim == 3 and sample_x.shape[0] == 1
    print(" - Dataset item retrieval verified.")

    # =========================================================================
    # 5. VERIFY TRAINING LOOP
    # =========================================================================
    print("\n[4/5] Running Training Demo...")

    # Run training for 1 epoch with small batch size
    # We use the cached data generated in the previous step
    train.train_model(
        num_epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        load_cached_data=True,
        max_samples=2,
    )

    assert os.path.exists(config.BEST_MODEL_PATH), "Model checkpoint was not created"
    print(f" - Training complete. Checkpoint saved to {config.BEST_MODEL_PATH}")

    # =========================================================================
    # 6. VERIFY INFERENCE PIPELINE
    # =========================================================================
    print("\n[5/5] Running Inference Demo...")

    if os.path.exists(config.TEST_METADATA_PATH):
        # Create a mini test set metadata file to avoid processing all test images
        df_test_full = pd.read_csv(config.TEST_METADATA_PATH)
        df_test_mini = df_test_full.head(2)  # Take first 2 images
        mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")
        df_test_mini.to_csv(mini_test_path, index=False)

        # Temporarily override test metadata path and disable TTA for speed
        original_test_path = config.TEST_METADATA_PATH
        original_tta = config.USE_TTA

        config.TEST_METADATA_PATH = mini_test_path
        config.USE_TTA = False

        try:
            predict.generate_submission(
                model_path=config.BEST_MODEL_PATH, output_path=config.SUBMISSION_FILE
            )
        finally:
            # Restore config
            config.TEST_METADATA_PATH = original_test_path
            config.USE_TTA = original_tta

        # Verify Submission File
        assert os.path.exists(config.SUBMISSION_FILE), "Submission file not created"

        df_sub = pd.read_csv(config.SUBMISSION_FILE)
        assert not df_sub.empty, "Submission file is empty"
        assert list(df_sub.columns) == ["id", "value"], "Incorrect submission columns"

        # Check value range
        vals = df_sub["value"].values
        assert (
            vals.min() >= 0.0 and vals.max() <= 1.0
        ), "Predictions out of range [0, 1]"

        print(f" - Submission generated successfully with {len(df_sub)} rows.")
    else:
        print(" - Test metadata not found, skipping inference verification.")

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
