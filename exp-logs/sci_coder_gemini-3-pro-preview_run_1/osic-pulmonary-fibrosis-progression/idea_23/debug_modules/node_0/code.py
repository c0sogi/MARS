import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_processing import (
    read_dicom_volume,
    generate_tri_slabs,
    process_patient,
)
from library.dataset import LungDataset
from library.model import ChannelAdaptiveDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.train_eval import train_model, set_seed


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("--- 1. Setup and Configuration ---")

    # Initialize directories
    Config.setup()

    # Override Config for rapid demonstration
    Config.EPOCHS = 1
    Config.DEBUG_DATA_SIZE = 10  # Use only 10 patients for training loop demo
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    set_seed(Config.SEED)
    print("Configuration initialized. Runtime parameters optimized for speed.")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # -------------------------------------------------------------------------
    # 2. Data Processing Verification
    # -------------------------------------------------------------------------
    print("\n--- 2. Data Processing Verification ---")

    # Select a sample patient
    sample_patient_id = train_df.iloc[0]["Patient"]
    sample_dicom_dir = os.path.join(Config.INPUT_DIR, train_df.iloc[0]["dicom_dir"])

    print(f"Processing sample patient: {sample_patient_id}")

    # A. Test Volume Reading
    volume = read_dicom_volume(sample_dicom_dir)
    assert volume is not None, "Failed to read DICOM volume."
    assert volume.ndim == 3, f"Volume should be 3D, got shape {volume.shape}"
    print(f"Volume read successfully. Shape: {volume.shape}")

    # B. Test Tri-Slab Generation (Axial)
    axial_img = generate_tri_slabs(volume, axis=0, img_size=Config.IMG_SIZE)
    assert axial_img.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), f"Axial image shape mismatch. Expected {(Config.IMG_SIZE, Config.IMG_SIZE, 3)}, got {axial_img.shape}"
    assert axial_img.dtype == np.uint8, "Image should be uint8."
    print("Axial Tri-Slab generated successfully.")

    # C. Test Full Patient Processing (with Caching)
    # This function generates both axial and coronal views and saves to cache
    ax_proc, cor_proc = process_patient(
        sample_patient_id, sample_dicom_dir, load_cached_data=False
    )
    assert ax_proc.shape == (Config.IMG_SIZE, Config.IMG_SIZE, 3)
    assert cor_proc.shape == (Config.IMG_SIZE, Config.IMG_SIZE, 3)

    # Verify cache files were created
    cache_ax_path = os.path.join(Config.CACHE_DIR, f"{sample_patient_id}_axial.npy")
    cache_cor_path = os.path.join(Config.CACHE_DIR, f"{sample_patient_id}_coronal.npy")
    assert os.path.exists(cache_ax_path), "Axial cache file not created."
    assert os.path.exists(cache_cor_path), "Coronal cache file not created."
    print("process_patient execution and caching verified.")

    # -------------------------------------------------------------------------
    # 3. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n--- 3. Dataset Verification ---")

    # Create a small subset dataset
    subset_df = train_df.head(4).copy()
    dataset = LungDataset(subset_df, mode="train")

    # Verify length
    assert len(dataset) == 4

    # Verify __getitem__
    sample = dataset[0]
    required_keys = [
        "axial",
        "coronal",
        "tabular",
        "fvc",
        "base_fvc",
        "week",
        "patient_id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify Tensor Shapes
    # Image: (3, H, W) because of ToTensorV2
    assert sample["axial"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected axial tensor shape: {sample['axial'].shape}"
    # Tabular: (6,)
    assert sample["tabular"].shape == (
        6,
    ), f"Unexpected tabular tensor shape: {sample['tabular'].shape}"

    print("Dataset item structure and tensor shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n--- 4. Model Architecture Verification ---")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = ChannelAdaptiveDualAxisNet().to(device)
    model.eval()

    # Create a dummy batch using DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    axial_in = batch["axial"].to(device)
    coronal_in = batch["coronal"].to(device)
    tabular_in = batch["tabular"].to(device)

    print(
        f"Input shapes - Axial: {axial_in.shape}, Coronal: {coronal_in.shape}, Tabular: {tabular_in.shape}"
    )

    with torch.no_grad():
        alpha, sigma_base, sigma_growth = model(axial_in, coronal_in, tabular_in)

    # Verify Output Shapes: Should be (Batch_Size,)
    assert alpha.shape == (2,), f"Alpha shape mismatch: {alpha.shape}"
    assert sigma_base.shape == (2,), f"Sigma Base shape mismatch: {sigma_base.shape}"
    assert sigma_growth.shape == (
        2,
    ), f"Sigma Growth shape mismatch: {sigma_growth.shape}"

    # Verify Constraints (Sigma must be positive due to Softplus)
    assert (sigma_base > 0).all(), "Sigma base must be positive."
    assert (sigma_growth > 0).all(), "Sigma growth must be positive."

    print("Model forward pass successful. Output constraints verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n--- 5. Loss Function Verification ---")

    criterion = LaplaceLogLikelihoodLoss()

    # Reconstruct predictions (Linear Trajectory)
    base_fvc = batch["base_fvc"].to(device)
    week_delta = batch["week"].to(device)
    target_fvc = batch["fvc"].to(device)

    pred_fvc = base_fvc + alpha * week_delta
    pred_sigma = sigma_base + sigma_growth * torch.abs(week_delta)

    loss = criterion(pred_fvc, pred_sigma, target_fvc)

    assert torch.isfinite(loss), "Loss is not finite (NaN or Inf)."
    assert loss.ndim == 0, "Loss should be a scalar."

    print(f"Loss calculation successful. Value: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 6. Training Loop Integration
    # -------------------------------------------------------------------------
    print("\n--- 6. Training Loop Integration (Debug Mode) ---")

    # We use the provided train_model function with debug=True.
    # This will use Config.DEBUG_DATA_SIZE (set to 10 above) and Config.EPOCHS (set to 1).
    best_model_path = train_model(train_df, val_df, debug=True)

    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    print(f"Training loop completed successfully. Model saved to: {best_model_path}")


if __name__ == "__main__":
    run_demo()
