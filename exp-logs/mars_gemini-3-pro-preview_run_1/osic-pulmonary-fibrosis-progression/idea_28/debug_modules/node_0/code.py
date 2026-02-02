import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything, LaplaceLogLikelihood
from library.dataset import FVCDataset, get_transforms
from library.model import VERNet
from library.engine import run_training, generate_submission_file


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    # 2. Data Pipeline Verification
    print("\n--- Verifying Data Pipeline ---")

    # Instantiate Dataset
    # We use 'train' mode which reads from ./metadata/train.csv
    train_dataset = FVCDataset(
        mode="train",
        transform=get_transforms("train"),
        data_dir="./input",
        cache_dir=os.path.join(working_dir, "cache_demo"),
    )

    print(f"Dataset size: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty"

    # Fetch one sample
    sample = train_dataset[0]
    print("Sample keys:", sample.keys())

    # Verify shapes
    # Images should be (3, 224, 224)
    assert sample["img_axial"].shape == (
        3,
        224,
        224,
    ), f"Axial shape mismatch: {sample['img_axial'].shape}"
    assert sample["img_coronal"].shape == (
        3,
        224,
        224,
    ), f"Coronal shape mismatch: {sample['img_coronal'].shape}"
    # Tabular vector should be length 6: [Age, Percent, Sex, Smoke_0, Smoke_1, Smoke_2]
    assert sample["tabular"].shape == (
        6,
    ), f"Tabular shape mismatch: {sample['tabular'].shape}"
    # Target should be scalar
    assert sample["target"].ndim == 0, "Target should be scalar"

    print("Single sample shapes verified.")

    # Verify DataLoader
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(train_loader))

    print(f"Batch shapes verified for batch_size={batch_size}")
    assert batch["img_axial"].shape == (batch_size, 3, 224, 224)
    assert batch["tabular"].shape == (batch_size, 6)

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")

    # Instantiate model
    # We use pretrained=False here to speed up initialization for the check
    model = VERNet(pretrained=False).to(device)
    model.eval()

    # Move batch to device
    img_ax = batch["img_axial"].to(device)
    img_cor = batch["img_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    week = batch["week"].to(device)
    base_fvc = batch["baseline_fvc"].to(device)

    # Forward pass
    with torch.no_grad():
        fvc_pred, confidence = model(img_ax, img_cor, tabular, week, base_fvc)

    print(f"Prediction shape: {fvc_pred.shape}")
    print(f"Confidence shape: {confidence.shape}")

    assert fvc_pred.shape == (batch_size,), "FVC prediction shape mismatch"
    assert confidence.shape == (batch_size,), "Confidence shape mismatch"

    # 4. Loss Function Verification
    print("\n--- Verifying Loss Function ---")

    criterion = LaplaceLogLikelihood()
    target = batch["target"].to(device)

    loss = criterion(fvc_pred, confidence, target)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.ndim == 0, "Loss should be a scalar"

    # 5. Training Loop Demonstration
    print("\n--- Running Training Loop (Demo) ---")

    # We run for 1 epoch with a small batch size to demonstrate functionality quickly
    save_path = os.path.join(working_dir, "demo_model.pth")

    best_metric = run_training(
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        device=device,
        save_path=save_path,
        patience=1,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead in demo
    )

    print(f"Training finished. Best Metric: {best_metric}")
    assert os.path.exists(save_path), "Model checkpoint was not created."

    # 6. Inference Demonstration
    print("\n--- Running Inference (Demo) ---")

    submission_path = os.path.join(working_dir, "demo_submission.csv")

    generate_submission_file(
        model_path=save_path,
        output_path=submission_path,
        device=device,
        batch_size=8,
        num_workers=0,
    )

    assert os.path.exists(submission_path), "Submission file was not created."

    # Validate submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission rows: {len(df_sub)}")
    print("Submission Head:")
    print(df_sub.head())

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column {col} in submission"

    # Check for valid values
    assert not df_sub["FVC"].isnull().any(), "NaN found in FVC predictions"
    assert (
        not df_sub["Confidence"].isnull().any()
    ), "NaN found in Confidence predictions"
    assert (
        df_sub["Confidence"] >= 70
    ).all(), "Confidence values should be clipped >= 70"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
