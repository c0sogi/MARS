import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data_loader import process_data, get_data_loaders
from library.model import CSPHN, CBAM
from library.trainer import train_one_epoch, validate, run_fold


def main():
    print("=== Starting Demonstration of Iceberg Classification Pipeline ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demo Isolation
    # ------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Override Config defaults to run quickly
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.N_FOLDS = 2  # Setup for 2 folds, we will only run fold 0
    Config.WORK_DIR = "./working/demo_execution"
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORK_DIR, "processed_data.npz")

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    print(f"    Working Directory: {Config.WORK_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Step 2a: Process Data
    # This reads raw JSONs, processes images, handles missing angles, and saves .npz
    data_dict = process_data(load_cached_data=False)

    # Validate processed data keys
    required_keys = ["X_train_all", "y_train_all", "inc_train_all", "X_test"]
    for key in required_keys:
        if key not in data_dict:
            raise AssertionError(f"Missing key {key} in processed data.")
    print("    Data processing successful. Keys verified.")

    # Step 2b: Data Loaders
    # Get loaders for Fold 0
    train_loader, val_loader, test_loader = get_data_loaders(
        fold=0, load_cached_data=True
    )

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    expected_img_shape = (Config.BATCH_SIZE, 3, 75, 75)
    if images.shape != expected_img_shape:
        raise AssertionError(
            f"Expected image shape {expected_img_shape}, got {images.shape}"
        )

    if angles.shape[0] != Config.BATCH_SIZE:
        raise AssertionError(
            f"Expected {Config.BATCH_SIZE} angles, got {angles.shape[0]}"
        )

    if labels.shape[0] != Config.BATCH_SIZE:
        raise AssertionError(
            f"Expected {Config.BATCH_SIZE} labels, got {labels.shape[0]}"
        )

    print("    Data Loader verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Instantiate Model
    model = CSPHN().to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    # Forward Pass
    outputs = model(images, angles)

    print(f"    Output Shape: {outputs.shape}")
    print(f"    Sample Output Value: {outputs[0].item():.4f}")

    # Assertions
    if outputs.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {outputs.shape}"
        )

    if (outputs < 0).any() or (outputs > 1).any():
        raise AssertionError(
            "Model outputs are not within valid probability range [0, 1]."
        )

    # Verify CBAM component individually
    print("    Verifying CBAM block...")
    cbam = CBAM(planes=32).to(Config.DEVICE)
    dummy_feat = torch.randn(Config.BATCH_SIZE, 32, 37, 37).to(Config.DEVICE)
    cbam_out = cbam(dummy_feat)
    if cbam_out.shape != dummy_feat.shape:
        raise AssertionError(
            f"CBAM changed tensor shape. Expected {dummy_feat.shape}, got {cbam_out.shape}"
        )

    print("    Model verification passed.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Component Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Training Components...")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch manually
    print("    Running manual train_one_epoch...")
    avg_train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE
    )
    print(f"    Avg Train Loss: {avg_train_loss:.6f}")

    if not np.isfinite(avg_train_loss):
        raise AssertionError("Training loss is not finite (NaN or Inf).")

    # Run validation manually
    print("    Running manual validate...")
    avg_val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
    print(f"    Avg Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.4f}")

    if not np.isfinite(avg_val_loss):
        raise AssertionError("Validation loss is not finite.")

    print("    Training component verification passed.")

    # ------------------------------------------------------------------------
    # 5. Full Fold Execution Verification
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Full Fold Execution (run_fold)...")

    # This will run the full loop defined in trainer.py (for 1 epoch as configured)
    # It handles scheduling, saving checkpoints, etc.
    best_loss = run_fold(fold_idx=0, load_cached_data=True)

    print(f"    Run Fold 0 completed. Best Loss: {best_loss:.6f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORK_DIR, "csphn_model_fold_0.pth")
    if not os.path.exists(checkpoint_path):
        raise AssertionError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"    Checkpoint confirmed at {checkpoint_path}")

    # ------------------------------------------------------------------------
    # 6. Inference / Submission Verification
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Inference from Checkpoint...")

    # Load the model from the checkpoint
    inference_model = CSPHN().to(Config.DEVICE)
    checkpoint = load_checkpoint(checkpoint_path, inference_model)
    print(f"    Loaded checkpoint from Epoch {checkpoint['epoch']}")

    inference_model.eval()

    # Get a batch from test loader
    test_imgs, test_angles, _ = next(iter(test_loader))
    test_imgs = test_imgs.to(Config.DEVICE)
    test_angles = test_angles.to(Config.DEVICE)

    with torch.no_grad():
        preds = inference_model(test_imgs, test_angles)

    print(f"    Test Predictions Shape: {preds.shape}")
    print(f"    First 3 Predictions: {preds[:3].flatten().cpu().numpy()}")

    if preds.shape[0] != test_imgs.shape[0]:
        raise AssertionError("Prediction batch size mismatch.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
