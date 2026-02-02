import os
import pandas as pd
import torch
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    mixup_data,
    mixup_criterion,
    get_score,
    apply_tta,
)
from library.dataset import SETIDataset
from library.model import SiameseConvNeXt
from library.engine import train_one_epoch, validate


def main():
    print("=== SETI Signal Detection Library Demo ===")

    # --- 1. Setup & Configuration Patching ---
    # We create mini metadata files to run the demo on a very small subset of data for speed.
    demo_dir = "./working/demo_files"
    os.makedirs(demo_dir, exist_ok=True)

    print("\n[1] Preparing Mini Datasets...")
    # Load original metadata and sample a few rows
    train_df_full = pd.read_csv("./metadata/train.csv")
    val_df_full = pd.read_csv("./metadata/val.csv")
    test_df_full = pd.read_csv("./metadata/test.csv")

    # Create mini subsets (e.g., 16 samples for train, 8 for val/test)
    train_mini_path = os.path.join(demo_dir, "train_mini.csv")
    val_mini_path = os.path.join(demo_dir, "val_mini.csv")
    test_mini_path = os.path.join(demo_dir, "test_mini.csv")

    train_df_full.head(16).to_csv(train_mini_path, index=False)
    val_df_full.head(8).to_csv(val_mini_path, index=False)
    test_df_full.head(8).to_csv(test_mini_path, index=False)

    # Patch the Config class to use these mini files and demo settings
    print("[2] Patching Configuration...")
    Config.TRAIN_METADATA = train_mini_path
    Config.VAL_METADATA = val_mini_path
    Config.TEST_METADATA = test_mini_path
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline
    Config.USE_FULL_DATA = True  # We control size via the CSV files
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print("    Configuration patched successfully.")

    # --- 2. Utils Demonstration ---
    print("\n[3] Testing Utils (library.utils)...")
    seed_everything(Config.SEED)

    # Test Mixup
    batch_size = 4
    channels = 6
    h, w = 288, 256
    dummy_x = torch.randn(batch_size, channels, h, w).to(Config.DEVICE)
    dummy_y = torch.randint(0, 2, (batch_size,)).float().to(Config.DEVICE)

    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_x, dummy_y, alpha=1.0, device=Config.DEVICE
    )

    assert mixed_x.shape == dummy_x.shape, "Mixup output shape mismatch"
    assert y_a.shape == dummy_y.shape, "Mixup target shape mismatch"
    print("    Mixup logic verified.")

    # Test Scoring (AUC)
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    score = get_score(y_true, y_pred)
    assert 0.0 <= score <= 1.0, "AUC score out of range"
    print(f"    Scoring function verified (AUC: {score})")

    # --- 3. Dataset Demonstration ---
    print("\n[4] Testing Dataset (library.dataset)...")
    train_ds = SETIDataset(mode="train")
    val_ds = SETIDataset(mode="val")

    assert len(train_ds) == 16, f"Expected 16 training samples, got {len(train_ds)}"
    assert len(val_ds) == 8, f"Expected 8 validation samples, got {len(val_ds)}"

    # Fetch one item to check transforms and padding
    img, target = train_ds[0]
    # Expected shape: (Channels, Freq_Padded, Time) -> (6, 288, 256)
    print(f"    Sample shape: {img.shape}")
    assert img.shape == (6, 288, 256), f"Unexpected image shape: {img.shape}"
    assert isinstance(target, torch.Tensor), "Target is not a tensor"
    print("    Dataset loading and transforms verified.")

    # --- 4. Model Demonstration ---
    print("\n[5] Testing Model (library.model)...")
    # Initialize model without pretrained weights
    model = SiameseConvNeXt(pretrained=False).to(Config.DEVICE)

    # Forward pass check with dummy data
    with torch.no_grad():
        output = model(dummy_x)

    print(f"    Model output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output (B, 1), got {output.shape}"
    print("    Model forward pass verified.")

    # --- 5. Engine Demonstration (Training Loop) ---
    print("\n[6] Testing Engine (library.engine)...")

    # Setup DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Run Train Step
    print("    Running training step...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.DEVICE
    )
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run Validation Step
    print("    Running validation step...")
    val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)
    print(f"    Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save model for inference test
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("    Model saved successfully.")

    # --- 6. Inference Demonstration ---
    print("\n[7] Testing Inference (TTA)...")

    # Reload model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    # Test TTA (Test Time Augmentation)
    # apply_tta takes (B, C, H, W) and returns averaged probabilities
    with torch.no_grad():
        probs = apply_tta(model, dummy_x, Config.DEVICE)

    print(f"    TTA Output shape: {probs.shape}")
    assert probs.shape == (batch_size, 1), "TTA output shape mismatch"
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of range [0, 1]"
    print("    TTA inference verified.")

    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    main()
