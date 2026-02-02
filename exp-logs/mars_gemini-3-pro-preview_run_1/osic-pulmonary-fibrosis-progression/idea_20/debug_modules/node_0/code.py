import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library components
from library.utils import seed_everything, get_device
from library.dicom_processing import generate_orthogonal_tri_slabs
from library.data import LungDataset, get_dataloaders
from library.model import ModalityAwareDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.train import train_model


def run_demo():
    print("Starting Demonstration...")

    # 1. Setup Environment
    # =========================================================================
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Ensure working directories exist
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 2. Verify DICOM Processing Logic
    # =========================================================================
    print("\n--- Verifying DICOM Processing ---")
    # Load train metadata to find a valid patient
    train_meta = pd.read_csv("./metadata/train.csv")
    sample_patient = train_meta.iloc[0]
    pid = sample_patient["Patient"]

    # dicom_dir in metadata is relative to input root (e.g., "train/ID...")
    # The library function expects the full path to the directory
    pdir = os.path.join("./input", sample_patient["dicom_dir"])

    print(f"Processing patient: {pid}")

    # Generate slabs (this will create cache in ./working/idea_20)
    # We force re-computation (load_cached_data=False) to verify the image generation logic
    images = generate_orthogonal_tri_slabs(pid, pdir, load_cached_data=False)

    # Assertions to ensure image validity
    assert "axial" in images and "coronal" in images, "Missing keys in image dict"
    axial_img = images["axial"]
    coronal_img = images["coronal"]

    # Check shapes: (Height, Width, Channels) -> (224, 224, 3)
    assert axial_img.shape == (224, 224, 3), f"Axial shape mismatch: {axial_img.shape}"
    assert coronal_img.shape == (
        224,
        224,
        3,
    ), f"Coronal shape mismatch: {coronal_img.shape}"

    # Check value range and type
    assert axial_img.dtype == np.float32, "Image dtype should be float32"
    assert (
        0.0 <= axial_img.min() and axial_img.max() <= 1.0
    ), "Image values out of range [0, 1]"

    print("DICOM processing verification passed.")

    # 3. Verify Dataset and DataLoader
    # =========================================================================
    print("\n--- Verifying Dataset & DataLoader ---")
    # Initialize dataset
    ds = LungDataset(mode="train", metadata_dir="./metadata")
    print(f"Dataset size: {len(ds)}")

    # Get one sample to verify __getitem__
    sample = ds[0]
    required_keys = [
        "img_axial",
        "img_coronal",
        "tabular",
        "meta",
        "target",
        "patient_week",
    ]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Check tensor shapes
    # Images are transformed to tensors (Channels, Height, Width)
    assert sample["img_axial"].shape == (3, 224, 224), "Axial tensor shape incorrect"
    assert sample["tabular"].shape == (7,), "Tabular feature shape incorrect"
    assert sample["meta"].shape == (3,), "Meta feature shape incorrect"
    assert sample["target"].shape == (1,), "Target shape incorrect"

    # Initialize Loaders to verify batching
    loaders = get_dataloaders(batch_size=4, num_workers=2, metadata_dir="./metadata")
    batch = next(iter(loaders["train"]))

    # Verify batch dimensions
    assert batch["img_axial"].shape == (4, 3, 224, 224), "Batch image shape incorrect"

    print("DataLoader verification passed.")

    # 4. Verify Model Architecture
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")
    model = ModalityAwareDualAxisNet(tabular_input_dim=7, embedding_dim=1280)
    model.to(device)
    model.eval()

    # Prepare inputs from the previously fetched batch
    img_axial = batch["img_axial"].to(device)
    img_coronal = batch["img_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    meta = batch["meta"].to(device)

    # Forward pass
    with torch.no_grad():
        pred_fvc, pred_sigma = model(img_axial, img_coronal, tabular, meta)

    # Check output shapes and validity
    assert pred_fvc.shape == (4,), f"Pred FVC shape mismatch: {pred_fvc.shape}"
    assert pred_sigma.shape == (4,), f"Pred Sigma shape mismatch: {pred_sigma.shape}"
    assert not torch.isnan(pred_fvc).any(), "NaN in FVC prediction"
    assert (
        pred_sigma >= 0
    ).all(), "Negative confidence values detected (should be softplus)"

    print("Model architecture verification passed.")

    # 5. Verify Loss Function
    # =========================================================================
    print("\n--- Verifying Loss Function ---")
    criterion = LaplaceLogLikelihoodLoss()
    target = batch["target"].to(device)

    # Compute loss on batch
    loss = criterion(pred_fvc, pred_sigma, target)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    # Manual calculation check for correctness
    # Metric formula: - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    # Loss is negative metric.
    t_fvc = torch.tensor([2000.0], device=device)
    p_fvc = torch.tensor([2000.0], device=device)  # Delta = 0
    p_sigma = torch.tensor([100.0], device=device)  # Sigma > 70 (no clipping)

    calc_loss = criterion(p_fvc, p_sigma, t_fvc)

    # Term 1: 0 (since delta is 0)
    # Term 2: ln(sqrt(2) * 100) = ln(141.421356)
    expected = torch.log(np.sqrt(2) * p_sigma)

    assert torch.isclose(calc_loss, expected, atol=1e-4), "Loss calculation mismatch"

    print("Loss function verification passed.")

    # 6. Run Training Pipeline (Integration Test)
    # =========================================================================
    print("\n--- Running Training Pipeline (Fast Mode) ---")
    # We run for 1 epoch with a small batch size to verify the loop and submission generation.
    # The train_model function saves to ./working/best_model.pth and ./submission/submission.csv

    try:
        train_model(
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            patience=1,
            num_workers=2,
            seed=42,
        )
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # 7. Verify Outputs
    # =========================================================================
    print("\n--- Verifying Outputs ---")

    # Check Model Checkpoint existence
    model_path = "./working/best_model.pth"
    assert os.path.exists(model_path), "Model checkpoint not found"

    # Check Submission File existence
    sub_path = "./submission/submission.csv"
    assert os.path.exists(sub_path), "Submission file not found"

    # Validate Submission Content
    df_sub = pd.read_csv(sub_path)
    print(f"Submission rows: {len(df_sub)}")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch: {df_sub.columns}"

    # Check row count (Test set has 1908 rows according to metadata analysis)
    test_meta = pd.read_csv("./metadata/test.csv")
    assert len(df_sub) == len(
        test_meta
    ), f"Submission row count {len(df_sub)} != Test meta {len(test_meta)}"

    # Check for nulls
    assert not df_sub["FVC"].isnull().any(), "Null FVC in submission"
    assert not df_sub["Confidence"].isnull().any(), "Null Confidence in submission"

    print("Output verification passed.")
    print("\nDemonstration Complete.")


if __name__ == "__main__":
    run_demo()
