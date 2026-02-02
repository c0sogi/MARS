import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data import load_dataset, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.calibration import run_calibration_phase
from library.production import train_full_fit, generate_submission


def demo_pipeline():
    print("=== Starting Iceberg Classification Pipeline Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    # We override the Config constants to ensure the demo runs within minutes
    # while still executing the complete logic paths.
    print("Configuring environment for fast execution...")
    set_seed(Config.SEED)

    # Reduce epochs and folds for demonstration
    Config.PHASE1_MAX_EPOCHS = 1  # Normally 50
    Config.PHASE1_PATIENCE = 1  # Short patience
    Config.NUM_FOLDS = 2  # Minimum for CV
    Config.SWA_EPOCHS = 1  # Short SWA
    Config.BATCH_SIZE = 16  # Smaller batch
    Config.NUM_WORKERS = 2  # Reduce overhead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading and Verification
    print("\n--- Step 1: Data Loading Verification ---")
    # Load training dataset (this handles caching internally)
    ds_train = load_dataset("train", load_cached_data=True)

    # Verify Dataset properties
    assert len(ds_train) > 0, "Training dataset is empty."
    sample_img, sample_ang, sample_lbl = ds_train[0]

    # Check shapes: Image should be (3, 224, 224) after transform, Label (1,)
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Angle: {sample_ang}")
    print(f"Sample Label: {sample_lbl}")

    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image dimensions."
    assert sample_lbl.shape == (1,), "Incorrect label shape."
    # Angle should be a scalar tensor
    assert isinstance(sample_ang, torch.Tensor), "Angle is not a tensor."

    # 3. Model Verification
    print("\n--- Step 2: Model Architecture Verification ---")
    model = IcebergResNet18().to(device)

    # Create dummy batch
    dummy_imgs = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_angs = torch.tensor([0.5, -0.5]).float().to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_imgs, dummy_angs)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (2, 1), "Model output shape mismatch (expected Bx1)."
    print("Model forward pass successful.")

    # 4. Phase 1: Calibration (Simplified)
    print("\n--- Step 3: Running Phase 1 (Calibration) ---")
    # This runs Stratified K-Fold to find optimal epochs
    # We patched Config.NUM_FOLDS=2 and MAX_EPOCHS=1 for speed
    e_conv = run_calibration_phase()

    print(f"Calibration returned E_conv: {e_conv}")
    assert e_conv > 0, "Calibration phase returned invalid convergence epoch."

    # 5. Phase 2: Production Training (Single Model Demo)
    print("\n--- Step 4: Running Phase 2 (Production Training) ---")

    # In a real run, we would loop 5 times. Here we demonstrate one full fit.
    # We need to aggregate train and val data as done in production.py
    ds_val = load_dataset("val", load_cached_data=True)

    full_images = np.concatenate([ds_train.images, ds_val.images], axis=0)
    full_angles = np.concatenate([ds_train.angles, ds_val.angles], axis=0)
    full_labels = np.concatenate([ds_train.labels, ds_val.labels], axis=0)

    full_dataset = IcebergDataset(
        full_images, full_angles, full_labels, transform=get_transforms("train")
    )

    full_loader = DataLoader(
        full_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Train one SWA model
    # We use model_idx=0
    checkpoint_path = train_full_fit(0, e_conv, full_loader, device)

    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Model successfully trained and saved to {checkpoint_path}")

    # 6. Submission Generation
    print("\n--- Step 5: Generating Submission ---")

    # We need to ensure test data is processed
    # load_dataset("test") will process and cache it if not present
    _ = load_dataset("test", load_cached_data=True)

    # Generate submission using the single trained model
    # (In production, we would pass a list of 5 paths)
    generate_submission([checkpoint_path], device)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print("Submission Head:")
    print(df_sub.head())

    assert len(df_sub) == 321, f"Submission has incorrect number of rows: {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing."
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
