import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import LungDataset
from library.model import DAVRNet
from library.engine import fit, predict, LaplaceLogLikelihoodLoss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demo...")

    # 1. Setup and Configuration Override for Speed
    # We modify the Config class attributes directly to run a fast demonstration
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Override Config for demo purposes
    Config.debug = True
    Config.debug_sample_size = 10  # Use only 10 samples
    Config.epochs = 2  # Run only 2 epochs
    Config.batch_size = 2  # Small batch size
    Config.num_workers = 0  # Avoid multiprocessing overhead for small data
    Config.backbone_pretrained = False  # Disable download for speed/offline safety
    Config.load_cached_data = False  # Force processing to demonstrate pipeline

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n--- Testing Data Pipeline ---")

    # Instantiate Datasets
    train_ds = LungDataset(mode="train", debug=True)
    val_ds = LungDataset(mode="val", debug=True)

    print(f"Train Dataset Length: {len(train_ds)}")
    print(f"Val Dataset Length: {len(val_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # Fetch a single batch to verify shapes and types
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = ["axial", "coronal", "tabular", "rel_week", "fvc", "baseline_fvc"]
    for key in expected_keys:
        assert key in batch, f"Missing key {key} in batch"

    # Verify Shapes
    # Axial/Coronal: (B, 3, 224, 224)
    assert batch["axial"].shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    )
    assert batch["coronal"].shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    )
    # Tabular: (B, 7)
    assert batch["tabular"].shape == (Config.batch_size, 7)

    print("Data loading and augmentation successful.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n--- Testing Model Architecture ---")

    model = DAVRNet()
    model.to(device)

    # Prepare inputs from the batch
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)

    # Run Forward Pass
    alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

    # Verify Output Shapes: (B, 1)
    assert alpha.shape == (Config.batch_size, 1), f"Alpha shape mismatch: {alpha.shape}"
    assert sigma_base.shape == (
        Config.batch_size,
        1,
    ), f"Sigma Base shape mismatch: {sigma_base.shape}"
    assert sigma_growth.shape == (
        Config.batch_size,
        1,
    ), f"Sigma Growth shape mismatch: {sigma_growth.shape}"

    # Verify Constraints (Sigma must be positive due to Softplus)
    assert torch.all(sigma_base > 0), "Sigma base contains non-positive values"
    assert torch.all(sigma_growth > 0), "Sigma growth contains non-positive values"

    print("Model forward pass successful.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n--- Testing Loss Function ---")

    loss_fn = LaplaceLogLikelihoodLoss()

    rel_week = batch["rel_week"].to(device).unsqueeze(1)
    true_fvc = batch["fvc"].to(device).unsqueeze(1)
    baseline_fvc = batch["baseline_fvc"].to(device).unsqueeze(1)

    loss = loss_fn(alpha, sigma_base, sigma_growth, baseline_fvc, rel_week, true_fvc)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    print(f"Loss calculation successful. Initial Loss: {loss.item():.4f}")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n--- Running Training Loop (Demo) ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.eta_min
    )

    # Run the training engine
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        patience=Config.patience,
    )

    # Verify model checkpoint was saved
    assert os.path.exists(Config.model_path), "Model checkpoint was not created."
    print("Training loop completed and model saved.")

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\n--- Running Inference ---")

    # Instantiate Test Dataset
    test_ds = LungDataset(mode="test", debug=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # Load best model
    model.load_state_dict(torch.load(Config.model_path, map_location=device))

    # Generate Predictions
    predict(model, test_loader, device)

    # Verify Submission File
    assert os.path.exists(Config.submission_path), "Submission file not found."

    df_sub = pd.read_csv(Config.submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {df_sub.columns}"

    # Check content types
    assert pd.api.types.is_numeric_dtype(df_sub["FVC"]), "FVC column is not numeric"
    assert pd.api.types.is_numeric_dtype(
        df_sub["Confidence"]
    ), "Confidence column is not numeric"

    print("Inference successful. Demo Complete.")


if __name__ == "__main__":
    run_demo()
