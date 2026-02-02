import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.data_utils import process_patient
from library.dataset import get_dataloaders, LungDataset
from library.model import H2DAN
from library.loss import LaplaceLogLikelihoodLoss
from library.train import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("H2-DAN Pipeline Demonstration")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 samples
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script
    Config.CACHE_DIR = "./working/demo_cache/"  # Temporary cache

    # Ensure cache dir exists
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Cache Dir:  {Config.CACHE_DIR}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device:     {device}")

    # ---------------------------------------------------------
    # 2. Data Processing Verification (Single Patient)
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Processing (process_patient)...")

    # Load metadata to find a valid patient
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    sample_row = train_meta.iloc[0]
    patient_id = sample_row["Patient"]
    dicom_rel_path = sample_row["dicom_dir"]
    dicom_full_path = os.path.join(Config.INPUT_ROOT, dicom_rel_path)

    print(f"    Processing patient: {patient_id}")
    print(f"    DICOM path: {dicom_full_path}")

    # Run processing
    axial, coronal = process_patient(
        patient_id, dicom_full_path, Config.CACHE_DIR, load_cached=False
    )

    # Assertions
    assert isinstance(axial, np.ndarray), "Axial output must be a numpy array"
    assert isinstance(coronal, np.ndarray), "Coronal output must be a numpy array"
    assert axial.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), f"Axial shape mismatch. Expected ({Config.IMG_SIZE}, {Config.IMG_SIZE}, 3), got {axial.shape}"
    assert coronal.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), f"Coronal shape mismatch. Expected ({Config.IMG_SIZE}, {Config.IMG_SIZE}, 3), got {coronal.shape}"

    print("    -> Image processing successful. Shapes verified.")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoaders...")

    # Create loaders
    train_loader, val_loader, scaler = get_dataloaders()

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = [
        "axial",
        "coronal",
        "deep_tab",
        "raw_tab",
        "delta_week",
        "target",
        "patient_id",
    ]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Verify shapes
    b_size = batch["axial"].shape[0]
    assert (
        b_size == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {b_size}"
    assert batch["axial"].shape == (b_size, 3, 240, 240)
    assert batch["raw_tab"].shape == (b_size, 2)  # FVC, Percent
    assert batch["target"].shape == (b_size,)

    print("    -> DataLoader produced valid batch structure.")

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying H2-DAN Model Architecture...")

    # Determine tabular input dimension from the batch
    tabular_dim = batch["deep_tab"].shape[1]
    print(f"    Tabular Input Dimension: {tabular_dim}")

    # Initialize model
    model = H2DAN(tabular_input_dim=tabular_dim)
    model.to(device)

    # Move batch to device
    batch_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Forward pass
    outputs = model(batch_device)

    # Verify outputs
    assert "fvc_pred" in outputs
    assert "confidence_pred" in outputs
    assert outputs["fvc_pred"].shape == (b_size,)
    assert outputs["confidence_pred"].shape == (b_size,)

    # Check for NaNs
    if torch.isnan(outputs["fvc_pred"]).any():
        raise ValueError("Model produced NaN predictions.")

    print("    -> Forward pass successful. Output shapes verified.")

    # ---------------------------------------------------------
    # 5. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Laplace Log Likelihood Loss...")

    criterion = LaplaceLogLikelihoodLoss()
    criterion.to(device)

    loss = criterion(
        outputs["fvc_pred"], outputs["confidence_pred"], batch_device["target"]
    )

    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"

    print(f"    -> Loss calculation successful. Value: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 6. Training Loop Simulation
    # ---------------------------------------------------------
    print("\n[6] Simulating Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train step
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.4f}")

    # Validation step
    val_metric = validate(model, val_loader, criterion, device)
    print(f"    Val Metric: {val_metric:.4f}")

    assert train_loss > -1000 and train_loss < 1000, "Train loss out of expected range"

    print("    -> Training and Validation steps completed without error.")

    # ---------------------------------------------------------
    # 7. Cleanup
    # ---------------------------------------------------------
    print("\n[7] Cleaning up...")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
        print(f"    Removed temporary cache: {Config.CACHE_DIR}")

    print("\n" + "=" * 50)
    print("DEMO COMPLETE: All components verified.")
    print("=" * 50)


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
