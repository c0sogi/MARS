import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model_factory as model_factory
import library.trainer as trainer


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("[1] Configuration & Setup")
    # Override config for speed
    config.BATCH_SIZE = 2
    config.NUM_EPOCHS = 1
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    utils.set_seed(42)
    print(f"    Device: {config.DEVICE}")
    print(f"    Batch Size: {config.BATCH_SIZE}")
    print("    Configuration overridden for demo speed.")

    # -------------------------------------------------------------------------
    # 2. Prepare Data Subsets
    # -------------------------------------------------------------------------
    print("\n[2] Preparing Data Subsets")
    # Load full metadata
    train_csv_path = os.path.join(config.METADATA_DIR, "train.csv")
    test_csv_path = os.path.join(config.METADATA_DIR, "test.csv")

    df_train_full = pd.read_csv(train_csv_path)
    df_test_full = pd.read_csv(test_csv_path)

    # Create tiny subsets (4 samples each) to ensure speed
    df_train_subset = df_train_full.head(4).copy()
    df_test_subset = df_test_full.head(4).copy()

    print(f"    Train subset size: {len(df_train_subset)}")
    print(f"    Test subset size: {len(df_test_subset)}")

    # -------------------------------------------------------------------------
    # 3. Test Utilities
    # -------------------------------------------------------------------------
    print("\n[3] Testing Utilities (library.utils)")

    # Test read_dicom_robust and resize_image
    # Pick a valid path from the subset
    sample_row = df_train_subset.iloc[0]
    flair_dir = os.path.join(config.INPUT_DIR, sample_row["path_FLAIR"])

    # Find a real dicom file
    dcm_files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]
    if dcm_files:
        dcm_path = os.path.join(flair_dir, dcm_files[0])
        print(f"    Testing read on: {dcm_files[0]}")

        # Read
        img = utils.read_dicom_robust(dcm_path)
        assert isinstance(
            img, np.ndarray
        ), "read_dicom_robust should return numpy array"
        print(f"    Read shape: {img.shape}")

        # Resize
        target_size = (224, 224)
        img_resized = utils.resize_image(img, target_size)
        assert img_resized.shape == (
            224,
            224,
        ), f"Resize failed. Expected {target_size}, got {img_resized.shape}"

        # Normalize
        img_norm = utils.normalize_minmax(img_resized)
        assert img_norm.dtype == np.float32, "Normalization should return float32"
        assert 0.0 <= img_norm.max() <= 1.0, "Max value should be <= 1.0"
        assert 0.0 <= img_norm.min() <= 1.0, "Min value should be >= 0.0"
        print("    Utility functions verified.")
    else:
        print("    Warning: No DICOM files found in sample directory to test utils.")

    # -------------------------------------------------------------------------
    # 4. Test Data Loader
    # -------------------------------------------------------------------------
    print("\n[4] Testing Data Loader (library.data_loader)")

    # Test get_dataloader (which handles anchor computation internally)
    # This will trigger compute_roi_anchors on the subset
    train_loader = data_loader.get_dataloader(
        df_train_subset,
        batch_size=config.BATCH_SIZE,
        phase="train",
        load_cached_anchors=False,  # Force compute for demo
    )

    print("    DataLoader initialized.")

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    # Verify shapes
    # Expected input: (Batch, Channels=20, H=224, W=224)
    expected_shape = (config.BATCH_SIZE, 20, 224, 224)
    assert (
        inputs.shape == expected_shape
    ), f"Input shape mismatch. Expected {expected_shape}, got {inputs.shape}"
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Target shape mismatch. Expected ({config.BATCH_SIZE},), got {targets.shape}"
    assert inputs.dtype == torch.float32, "Input tensor should be float32"

    print(f"    Batch Input Shape: {inputs.shape}")
    print(f"    Batch Target Shape: {targets.shape}")
    print("    Data Loader verified.")

    # -------------------------------------------------------------------------
    # 5. Test Model Architecture
    # -------------------------------------------------------------------------
    print("\n[5] Testing Model (library.model_factory)")

    model = model_factory.AsymmetricEfficientNet()
    model.to(config.DEVICE)

    # Pass the batch through the model
    inputs = inputs.to(config.DEVICE)
    outputs = model(inputs)

    # Expected output: (Batch, 1) - Logits
    assert outputs.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(config.BATCH_SIZE, 1)}, got {outputs.shape}"

    print(f"    Model Output Shape: {outputs.shape}")
    print("    Model architecture verified.")

    # -------------------------------------------------------------------------
    # 6. Test Training Loop
    # -------------------------------------------------------------------------
    print("\n[6] Testing Training Loop (library.trainer)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch of training using the subset loader
    print("    Running train_one_epoch...")
    train_loss, train_auc = trainer.train_one_epoch(
        model, train_loader, criterion, optimizer, config.DEVICE
    )

    print(f"    Train Loss: {train_loss:.4f}")
    print(f"    Train AUC:  {train_auc:.4f}")

    # Run validation (using the same subset for demo purposes)
    print("    Running validate...")
    val_loss, val_auc = trainer.validate(model, train_loader, criterion, config.DEVICE)
    print(f"    Val Loss:   {val_loss:.4f}")

    # Save a temporary model for inference test
    temp_model_path = os.path.join(config.WORKING_DIR, "demo_model.pth")
    torch.save(model.state_dict(), temp_model_path)
    print(f"    Model saved to {temp_model_path}")

    # -------------------------------------------------------------------------
    # 7. Test Inference Logic (TTA)
    # -------------------------------------------------------------------------
    print("\n[7] Testing Inference Logic")

    # Create test loader
    test_loader = data_loader.get_dataloader(
        df_test_subset,
        batch_size=config.BATCH_SIZE,
        phase="test",
        load_cached_anchors=False,
    )

    # Load model
    model.eval()
    predictions = []

    print("    Running inference with TTA...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(config.DEVICE)

            # Simulate TTA steps from trainer.predict_and_submit
            # 1. Original
            out_orig = torch.sigmoid(model(inputs))

            # 2. HFlip
            inputs_h = torch.flip(inputs, [-1])  # Simple flip for demo
            out_h = torch.sigmoid(model(inputs_h))

            # 3. VFlip
            inputs_v = torch.flip(inputs, [-2])
            out_v = torch.sigmoid(model(inputs_v))

            # Average
            avg_preds = (out_orig + out_h + out_v) / 3.0
            predictions.extend(avg_preds.cpu().numpy().flatten())

    assert len(predictions) == len(df_test_subset), "Prediction count mismatch"
    print(f"    Generated {len(predictions)} predictions.")
    print(f"    Sample predictions: {predictions[:2]}")

    # Create submission dataframe
    sub_df = df_test_subset[["BraTS21ID"]].copy()
    sub_df["MGMT_value"] = predictions

    demo_sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(demo_sub_path, index=False)
    print(f"    Demo submission saved to {demo_sub_path}")

    print("\n=== Library Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
