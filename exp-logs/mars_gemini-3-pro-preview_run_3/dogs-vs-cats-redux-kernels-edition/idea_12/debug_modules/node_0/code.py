import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.custom_layers import GeM, MultiSampleDropout
from library.data_loader import get_dataloaders, get_test_dataloader
from library.model_factory import build_model
from library.trainer import Trainer


def run_demonstration():
    print("=== Starting Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("[1] Configuring environment...")

    # Set reproducible seeds
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 images
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead for small data

    # Use a temporary working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean/Create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print("    Configuration updated for speed (Debug Mode: ON).")

    # -------------------------------------------------------------------------
    # 2. Verify Custom Layers
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Custom Layers...")

    # Test GeM Pooling
    # Input: (Batch, Channels, Height, Width) -> Output: (Batch, Channels, 1, 1)
    dummy_input = torch.randn(4, 2048, 16, 16)
    gem_layer = GeM(p=3.0)
    gem_output = gem_layer(dummy_input)

    assert gem_output.shape == (
        4,
        2048,
        1,
        1,
    ), f"GeM output shape mismatch. Expected (4, 2048, 1, 1), got {gem_output.shape}"
    print("    GeM Pooling: Passed (Shape verified).")

    # Test Multi-Sample Dropout
    # Input: (Batch, Features) -> Output: (Batch, Classes)
    msd_input = torch.randn(4, 512)
    msd_layer = MultiSampleDropout(in_features=512, out_features=1, num_samples=3)
    msd_output = msd_layer(msd_input)

    assert msd_output.shape == (
        4,
        1,
    ), f"MSD output shape mismatch. Expected (4, 1), got {msd_output.shape}"
    print("    Multi-Sample Dropout: Passed (Shape verified).")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loaders
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loaders...")

    # Initialize Loaders
    train_loader, val_loader = get_dataloaders(
        resolution=224, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )
    test_loader = get_test_dataloader(
        resolution=224, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Check Train Loader
    images, labels = next(iter(train_loader))
    print(f"    Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Train image batch shape incorrect."
    assert labels.shape == (Config.BATCH_SIZE,), "Train label batch shape incorrect."

    # Check Test Loader
    test_images, test_ids = next(iter(test_loader))
    print(f"    Test Batch  - Images: {test_images.shape}, IDs: {test_ids.shape}")

    assert test_images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Test image batch shape incorrect."
    print("    Data Loaders: Passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Construction
    # -------------------------------------------------------------------------
    print("\n[4] Building Model...")

    # Use a lighter backbone (resnet18) for the demo to save memory/time
    # Enable GeM to test the custom pooling integration
    model = build_model(
        model_name="resnet18",
        num_classes=1,
        pretrained=True,
        use_gem=True,
        use_msd=False,
    )
    model.to(device)

    # Verify Forward Pass
    dummy_batch = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_batch)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        1,
    ), f"Model output shape incorrect. Expected (2, 1), got {output.shape}"
    print("    Model Construction: Passed.")

    # -------------------------------------------------------------------------
    # 5. Execute Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path=os.path.join(Config.WORKING_DIR, "demo_model.pth"),
    )

    # Run Fit
    # Note: We use the small debug loaders created in step 3
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Check if model checkpoint was saved
    assert os.path.exists(trainer.save_path), "Model checkpoint was not saved."
    print("    Training Loop: Passed (Checkpoint saved).")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    trainer.generate_submission(test_loader, output_path=Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(df_sub)}")
    print(f"    Submission Columns: {list(df_sub.columns)}")

    # Logic checks
    assert (
        "id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."
    assert (
        df_sub["label"].min() >= 0 and df_sub["label"].max() <= 1
    ), "Probabilities out of range."

    print("    Submission Generation: Passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
