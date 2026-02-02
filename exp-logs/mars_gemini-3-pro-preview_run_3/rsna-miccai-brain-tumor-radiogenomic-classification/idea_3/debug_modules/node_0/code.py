import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.utils import seed_everything, get_device
from library.model import BraTS25DEfficientNet
from library.data_loader import BraTSDataset
from library.train import train_one_epoch, validate, predict_and_submit


def run_demo():
    print("=== Starting Library Verification Demo ===")

    # 1. Setup Environment
    print("\n[1] Setting up environment...")
    seed_everything(42)
    device = get_device()
    print(f"    Device: {device}")

    # 2. Verify Model Architecture
    print("\n[2] Verifying Model (BraTS25DEfficientNet)...")
    # Use pretrained=False to avoid downloading weights during demo
    model = BraTS25DEfficientNet(pretrained=False).to(device)

    # Create dummy input: Batch=2, Channels=64 (16 slices * 4 mods), H=256, W=256
    # Note: The model expects 64 channels as input.
    dummy_input = torch.randn(2, 64, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")

    # Assert output is (Batch, 1)
    if output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
        )
    print("    Model verification passed.")

    # 3. Verify Dataset Logic
    print("\n[3] Verifying Dataset (BraTSDataset)...")
    # Constants matching library/data_loader.py
    IMG_SIZE = 256
    NUM_SAMPLES = 4
    NUM_MODALITIES = 4
    NUM_SLICES = 32

    # Create synthetic data: (N, 4, 32, 256, 256)
    print("    Generating synthetic dataset (approx 300MB)...")
    X_synthetic = np.random.rand(
        NUM_SAMPLES, NUM_MODALITIES, NUM_SLICES, IMG_SIZE, IMG_SIZE
    ).astype(np.float32)
    y_synthetic = np.random.randint(0, 2, size=(NUM_SAMPLES,)).astype(np.float32)
    ids_synthetic = np.array([f"{i:05d}" for i in range(NUM_SAMPLES)])

    # Test Train Mode (Subsampling)
    train_ds = BraTSDataset(X_synthetic, y_synthetic, ids_synthetic, mode="train")
    sample_x, sample_y = train_ds[0]

    print(f"    Train item shape: {sample_x.shape}")
    # Expected: (64, 256, 256) -> 4 modalities * 16 slices
    if sample_x.shape != (64, IMG_SIZE, IMG_SIZE):
        raise AssertionError(
            f"Train dataset shape mismatch. Expected (64, 256, 256), got {sample_x.shape}"
        )

    # Test Val Mode (Full Volume)
    val_ds = BraTSDataset(X_synthetic, y_synthetic, ids_synthetic, mode="val")
    sample_x_val, sample_y_val = val_ds[0]

    print(f"    Val item shape:   {sample_x_val.shape}")
    # Expected: (128, 256, 256) -> 4 modalities * 32 slices
    if sample_x_val.shape != (128, IMG_SIZE, IMG_SIZE):
        raise AssertionError(
            f"Val dataset shape mismatch. Expected (128, 256, 256), got {sample_x_val.shape}"
        )
    print("    Dataset verification passed.")

    # 4. Verify Training Loop Components
    print("\n[4] Verifying Training & Validation Steps...")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Test Train Step
    print("    Running train_one_epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.4f}")

    if np.isnan(train_loss):
        raise AssertionError("Train loss returned NaN.")

    # Test Validation Step
    print("    Running validate...")
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    if np.isnan(val_loss):
        raise AssertionError("Validation loss returned NaN.")
    print("    Training loop verification passed.")

    # 5. Verify Inference & Submission
    print("\n[5] Verifying Inference & Submission...")

    # Create Test Dataset (No labels)
    test_ds = BraTSDataset(X_synthetic, None, ids_synthetic, mode="test")
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    output_path = "./working/demo_submission.csv"

    # Run prediction logic
    predict_and_submit(model, test_loader, ids_synthetic, device, output_path)

    # Check if file exists
    if not os.path.exists(output_path):
        raise AssertionError(f"Submission file was not created at {output_path}")

    # Check content
    df_sub = pd.read_csv(output_path)
    print(f"    Submission file created with {len(df_sub)} rows.")
    print(df_sub.head())

    if len(df_sub) != NUM_SAMPLES:
        raise AssertionError(
            f"Submission row count mismatch. Expected {NUM_SAMPLES}, got {len(df_sub)}"
        )

    if list(df_sub.columns) != ["BraTS21ID", "MGMT_value"]:
        raise AssertionError(f"Submission columns mismatch. Got {list(df_sub.columns)}")

    print("    Inference verification passed.")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
