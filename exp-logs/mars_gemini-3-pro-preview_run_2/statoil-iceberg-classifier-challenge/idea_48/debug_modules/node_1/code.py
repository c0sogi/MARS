import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import DEVICE, BATCH_SIZE
from library.utils import seed_everything
from library.layers import DualPooling, CBAM, WideConvBlock
from library.model import DCWBN, DilatedReadout
from library.data_loader import load_and_process_data, IcebergDataset
from library.train import train_one_epoch, validate


def run_demo():
    print("----------------------------------------------------------------")
    print("  Iceberg Classifier Library Demo")
    print("----------------------------------------------------------------")

    # 1. Setup
    seed_everything(42)
    print("[Setup] Seed set to 42.")

    # 2. Data Loading & Processing
    print("\n[Data] Loading and processing data...")
    # This function handles loading JSONs, processing bands, and caching.
    # We use the real data but will subset it immediately for the demo.
    X_train, y_train, inc_train, train_ids, X_test, inc_test, test_ids = (
        load_and_process_data(load_cached_data=True)
    )

    print(f"  Original Train Shape: {X_train.shape}")
    print(f"  Original Test Shape:  {X_test.shape}")

    # Verify Data Integrity
    assert len(X_train) == len(y_train) == len(inc_train)
    assert X_train.shape[1] == 3  # 3 Channels
    assert X_train.shape[2] == 75  # Height
    assert X_train.shape[3] == 75  # Width

    # Create a tiny subset for speed (32 samples = 2 batches of 16)
    subset_size = 32
    demo_batch_size = 16

    X_mini = X_train[:subset_size]
    y_mini = y_train[:subset_size]
    inc_mini = inc_train[:subset_size]

    print(f"  Created subset of size {subset_size} for demonstration.")

    # Create Dataset and Loader
    train_ds = IcebergDataset(X_mini, inc_mini, y_mini, transform=True)
    val_ds = IcebergDataset(
        X_mini, inc_mini, y_mini, transform=False
    )  # Use same for val to keep it simple

    train_loader = DataLoader(train_ds, batch_size=demo_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=demo_batch_size, shuffle=False)

    # 3. Layer Verification
    print("\n[Layers] Verifying custom layers...")

    # Test DualPooling
    # Input: (B, C, H, W) -> Output: (B, 2C, H/2, W/2)
    dummy_input = torch.randn(4, 64, 75, 75)
    pool = DualPooling(kernel_size=2, stride=2)
    out_pool = pool(dummy_input)
    print(f"  DualPooling Output: {out_pool.shape}")
    assert out_pool.shape == (4, 128, 37, 37), "DualPooling shape mismatch"

    # Test CBAM
    # Input: (B, C, H, W) -> Output: (B, C, H, W)
    cbam = CBAM(in_planes=64)
    out_cbam = cbam(dummy_input)
    print(f"  CBAM Output: {out_cbam.shape}")
    assert out_cbam.shape == (4, 64, 75, 75), "CBAM shape mismatch"

    # Test WideConvBlock
    # Input: (B, C, H, W) -> Output: (B, OutC, H, W)
    conv = WideConvBlock(in_channels=64, out_channels=128)
    out_conv = conv(dummy_input)
    print(f"  WideConvBlock Output: {out_conv.shape}")
    assert out_conv.shape == (4, 128, 75, 75), "WideConvBlock shape mismatch"

    # 4. Model Instantiation
    print("\n[Model] Instantiating DCWBN...")
    model = DCWBN().to(DEVICE)

    # Test Forward Pass
    dummy_img = torch.randn(2, 3, 75, 75).to(DEVICE)
    dummy_inc = torch.tensor([35.0, 40.0]).float().unsqueeze(1).to(DEVICE)  # (B, 1)

    with torch.no_grad():
        logits = model(dummy_img, dummy_inc)

    print(f"  Model Output Logits: {logits.shape}")
    assert logits.shape == (2, 2), "Model output shape mismatch (expected [Batch, 2])"

    # 5. Training Loop Demo
    print("\n[Training] Running training loop demo (1 Epoch)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train 1 Epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0 <= train_acc <= 1, "Training accuracy out of bounds"

    # Validate
    print("[Validation] Running validation...")
    val_loss, val_acc, val_preds, val_targets = validate(
        model, val_loader, criterion, DEVICE
    )
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert len(val_preds) == subset_size, "Validation predictions count mismatch"
    assert (
        val_preds.min() >= 0 and val_preds.max() <= 1
    ), "Predictions are not valid probabilities"

    print("\n[Success] All components verified successfully.")


if __name__ == "__main__":
    run_demo()
