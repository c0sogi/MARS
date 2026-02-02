import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, dice_coef_metric
from library.dataset import ContrailDataset
from library.network import AttentionGatedUNet
from library.losses import HybridLoss
from library.engine import train_model
from library.inference import predict_and_submit


def run_demo():
    print("=" * 50)
    print("Starting Contrail Identification Demo")
    print("=" * 50)

    # 1. Setup and Configuration
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Set seeds for reproducibility
    Config.set_seed(42)

    # Override Config parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.MAX_TRAIN_SAMPLES = 64  # Small subset for training
    Config.MAX_VAL_SAMPLES = 32  # Small subset for validation
    Config.NUM_WORKERS = 2  # Reduce overhead

    # Disable pretrained weights download for speed/offline safety in demo
    Config.ENCODER_WEIGHTS = None

    # Ensure clean working directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()

    print(f"    Device: {Config.DEVICE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Train Samples: {Config.MAX_TRAIN_SAMPLES}")

    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a simple 4x4 mask:
    # 0 1 1 0
    # 0 1 0 0
    # ...
    # Flattened (column-major/Fortran): 0,0,.., 1,1,.., 1,0..
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0, 1] = 1  # Pixel 5 (1-based column-major)
    dummy_mask[1, 1] = 1  # Pixel 6
    # Flattened indices for column 1 (indices 4-7 in 0-based): 0, 1, 1, 0 -> pixels 5, 6 are set.
    # rle_encode expects 1-based indexing.
    # In Fortran order: col0 (1-4), col1 (5-8), col2 (9-12), col3 (13-16)
    # Mask at (0,1) is index 5. Mask at (1,1) is index 6.
    # Expected RLE: "5 2" (Start at 5, length 2)
    encoded = rle_encode(dummy_mask)
    assert encoded == "5 2", f"RLE Encoding failed. Expected '5 2', got '{encoded}'"
    print("    RLE Encoding: OK")

    # Test Dice Metric
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y_pred = torch.tensor([1, 0, 0, 0], dtype=torch.float32)  # 0.5 overlap
    # Intersection = 1, Union = 1 + 2 = 3. Dice = 2*1 / 3 = 0.666...
    dice = dice_coef_metric(y_pred, y_true, threshold=0.5)
    assert (
        abs(dice - 0.666666) < 1e-4
    ), f"Dice Metric failed. Expected ~0.666, got {dice}"
    print("    Dice Metric: OK")

    # 3. Data Loading
    # ---------------------------------------------------------
    print("\n[3] Initializing Datasets and DataLoaders...")

    train_dataset = ContrailDataset(split="train", max_samples=Config.MAX_TRAIN_SAMPLES)
    val_dataset = ContrailDataset(
        split="validation", max_samples=Config.MAX_VAL_SAMPLES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify batch shape
    images, masks = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Mask Shape: {masks.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image tensor shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect mask tensor shape"
    print("    Data Loading: OK")

    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing Attention-Gated U-Net...")

    model = AttentionGatedUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(
            2, Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE
        ).to(Config.DEVICE)
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        2,
        Config.NUM_CLASSES,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Model output shape mismatch. Expected (2, 1, 256, 256), got {dummy_output.shape}"
    print("    Model Forward Pass: OK")

    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Starting Training Loop...")

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    best_score = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_epochs=Config.EPOCHS,
        patience=2,
    )

    print(f"    Training completed. Best Validation Dice: {best_score:.4f}")
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "best_model.pth")
    ), "best_model.pth was not saved."

    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Run inference on a small subset of test data for demonstration
    predict_and_submit(
        checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        max_samples=20,  # Only predict on 20 test samples
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "record_id" in df_sub.columns and "encoded_pixels" in df_sub.columns
    ), "Submission file missing required columns."
    print("    Submission generated successfully.")
    print(f"    Submission head:\n{df_sub.head()}")

    print("\n" + "=" * 50)
    print("Demo Completed Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
