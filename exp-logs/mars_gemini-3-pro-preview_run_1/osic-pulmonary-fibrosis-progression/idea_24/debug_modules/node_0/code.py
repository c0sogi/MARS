import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import OSICDataset, get_transforms, get_dataloaders
from library.model import TMIGN, LaplaceLogLikelihoodLoss
from library.train import run_training, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Reproducibility
    print("\n[1] Setting up environment...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # Ensure working directories exist (handled by Config.setup, but good to double check logic)
    assert os.path.exists(Config.WORKING_DIR), "Working directory not created."

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline (OSICDataset & TriSlabGenerator)...")

    # Load a small subset of training metadata for testing
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Metadata file not found: {train_csv_path}")

    df_train = pd.read_csv(train_csv_path).head(4)  # Use only 4 samples

    # Initialize Dataset
    dataset = OSICDataset(
        df=df_train,
        cache_dir=Config.CACHE_DIR,
        mode="train",
        transform=get_transforms("train"),
    )

    print(f"    Dataset length: {len(dataset)}")
    assert len(dataset) == 4, "Dataset length mismatch."

    # Fetch one sample to verify structure and shapes
    sample = dataset[0]
    required_keys = [
        "img_ax",
        "img_cor",
        "tabular",
        "weeks",
        "fvc",
        "base_fvc",
        "base_week",
        "patient_id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify Image Shapes (Channels, Height, Width) -> (3, 224, 224)
    img_shape = sample["img_ax"].shape
    assert img_shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_shape}"

    # Verify Tabular Shape (6 features)
    tab_shape = sample["tabular"].shape
    assert tab_shape == (6,), f"Incorrect tabular shape: {tab_shape}"

    print("    Data loading and preprocessing: OK")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (TMIGN)...")

    model = TMIGN().to(device)
    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_tab = torch.randn(batch_size, 6).to(device)

    # Forward pass
    with torch.no_grad():
        alpha, sigma_base, sigma_growth = model(dummy_img, dummy_img, dummy_tab)

    # Verify output shapes
    assert alpha.shape == (batch_size,), f"Alpha shape mismatch: {alpha.shape}"
    assert sigma_base.shape == (
        batch_size,
    ), f"Sigma_base shape mismatch: {sigma_base.shape}"
    assert sigma_growth.shape == (
        batch_size,
    ), f"Sigma_growth shape mismatch: {sigma_growth.shape}"

    # Verify positivity constraints (Softplus)
    assert (sigma_base >= 0).all(), "Sigma_base contains negative values."
    assert (sigma_growth >= 0).all(), "Sigma_growth contains negative values."

    print("    Model forward pass: OK")

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function (LaplaceLogLikelihoodLoss)...")

    criterion = LaplaceLogLikelihoodLoss()

    # Dummy predictions and targets
    pred_fvc = torch.tensor([2000.0, 2500.0], device=device)
    true_fvc = torch.tensor([2100.0, 2400.0], device=device)
    pred_sigma = torch.tensor([100.0, 150.0], device=device)

    loss = criterion(pred_fvc, true_fvc, pred_sigma)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert not torch.isnan(loss), "Loss is NaN."

    print(f"    Calculated Loss: {loss.item():.4f}")
    print("    Loss function: OK")

    # 5. Full Training Loop Verification
    print("\n[5] Running Training Loop (Debug Mode)...")

    # We use the provided run_training function with debug=True to use a subset of data
    # and limit epochs to 1 for speed.
    try:
        run_training(epochs=1, batch_size=4, patience=1, learning_rate=1e-4, debug=True)
    except Exception as e:
        print(f"    Training failed with error: {e}")
        raise e

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file (best_model.pth) was not created."
    print("    Training loop completed successfully.")

    # 6. Submission Generation Verification
    print("\n[6] Generating Submission...")

    try:
        generate_submission(batch_size=4)
    except Exception as e:
        print(f"    Submission generation failed with error: {e}")
        raise e

    # Check if submission file was created
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content format
    sub_df = pd.read_csv(submission_path)
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Submission missing column: {col}"

    assert len(sub_df) > 0, "Submission file is empty."
    print(f"    Submission generated with {len(sub_df)} rows.")
    print("    Submission generation: OK")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
