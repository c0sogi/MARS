import os
import torch
import torch.nn as nn
import pandas as pd
import shutil
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train import train_one_epoch, validate
from library.evaluate import generate_submission

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n=== Setting up Demo Configuration ===")

    # Override Config parameters for a fast demonstration
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch size for demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Data Loading ===")

    # Initialize DataLoaders
    # We set load_cached_data=False to force the ROI generation logic to run once for demo purposes,
    # or True if we want to rely on existing cache. Given the demo environment, we let it generate
    # in the new working dir.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify Batch Structure
    images, targets = next(iter(train_loader))

    # Expected: (Batch, Channels, Height, Width) -> (B, 12, 224, 224)
    expected_channels = Config.TOTAL_CHANNELS  # 12
    expected_size = Config.IMG_SIZE[0]  # 224

    print(f"Sample Batch Shape: {images.shape}")
    print(f"Sample Target Shape: {targets.shape}")

    assert images.dim() == 4, "Images must be 4D tensor"
    assert (
        images.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels (4 modalities * 3 slices)"
    assert (
        images.shape[2] == expected_size and images.shape[3] == expected_size
    ), "Image dimensions mismatch"
    assert (
        targets.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and targets"

    print("Data Loading Verification: PASSED")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Demonstration
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Model Architecture ===")

    model = AsymmetricEfficientNet().to(device)

    # Verify Input Stem Modification
    # The first layer should have 12 input channels and 4 groups
    first_conv = model.backbone.features[0][0]
    print(f"First Conv Layer: {first_conv}")

    assert first_conv.in_channels == expected_channels, "Model input channels mismatch"
    assert (
        first_conv.groups == 4
    ), "Model should use grouped convolutions for modalities"

    # Verify Forward Pass
    dummy_input = torch.randn(2, expected_channels, expected_size, expected_size).to(
        device
    )
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Dummy Output Shape: {output.shape}")
    assert output.shape == (
        2,
        1,
    ), "Output shape should be (Batch, 1) for binary classification"

    print("Model Architecture Verification: PASSED")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Training Loop (1 Epoch) ===")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    train_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, device, epoch=1
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    val_loss, val_auc = validate(val_loader, model, criterion, device)
    print(f"Epoch 1 Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Save checkpoint manually (simulating the save in run_training)
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": val_loss,
        "val_auc": val_auc,
    }
    torch.save(state, Config.MODEL_CHECKPOINT_PATH)
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Checkpoint file was not created"

    print("Training Loop Verification: PASSED")

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Inference & Submission ===")

    # Generate submission using the checkpoint we just saved
    # We use load_cached_data=True to reuse the ROI cache generated in step 2
    generate_submission(load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Check Columns
    assert "BraTS21ID" in df_sub.columns, "BraTS21ID column missing"
    assert "MGMT_value" in df_sub.columns, "MGMT_value column missing"

    # Check Row Count (Should match test metadata)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count mismatch. Expected {len(df_test_meta)}, got {len(df_sub)}"

    # Check Value Range
    probs = df_sub["MGMT_value"].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities must be between 0 and 1"

    print("Inference Verification: PASSED")
    print("\nAll demonstrations completed successfully.")
