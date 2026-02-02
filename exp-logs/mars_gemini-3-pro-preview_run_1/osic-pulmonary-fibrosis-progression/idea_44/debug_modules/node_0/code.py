import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SLADAN
from library.engine import Engine, LaplaceLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=" * 40)
    print("PULMONARY FIBROSIS MODEL DEMONSTRATION")
    print("=" * 40)

    # 1. Setup and Configuration
    # Seed for reproducibility
    seed_everything(42)

    # Initialize Config with debug=True to use a subset of data and limit epochs
    print("\n[1] Initializing Configuration...")
    cfg = Config(debug=True, epochs=1)
    cfg.display()

    # 2. Data Pipeline Verification
    print("[2] Verifying Data Pipeline...")
    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches:   {len(val_loader)}")
    print(f"  Test Batches:  {len(test_loader)}")

    # Fetch a single batch to inspect structure
    batch = next(iter(train_loader))
    img_ax = batch["img_axial"]
    img_cor = batch["img_coronal"]
    tab = batch["tabular"]
    target = batch["target"]
    week = batch["week"]
    pids = batch["patient_id"]

    print(f"  Batch Keys: {list(batch.keys())}")
    print(f"  Axial Image Shape:   {img_ax.shape} (Expected: [B, 3, 224, 224])")
    print(f"  Coronal Image Shape: {img_cor.shape} (Expected: [B, 3, 224, 224])")
    print(f"  Tabular Data Shape:  {tab.shape} (Expected: [B, 9])")
    print(f"  Target Shape:        {target.shape}")

    # Assertions to ensure data integrity
    assert img_ax.ndim == 4, "Axial images must be 4D tensors"
    assert img_cor.ndim == 4, "Coronal images must be 4D tensors"
    assert tab.shape[1] == 9, "Tabular data must have 9 features"
    assert not torch.isnan(tab).any(), "Tabular data contains NaNs"

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (SLADAN)...")
    device = cfg.DEVICE
    model = SLADAN(cfg).to(device)

    # Move batch to device
    img_ax = img_ax.to(device)
    img_cor = img_cor.to(device)
    tab = tab.to(device)

    # Perform Forward Pass
    preds = model(img_ax, img_cor, tab)
    print(f"  Model Output Shape: {preds.shape} (Expected: [B, 3])")

    # Assertions for model output
    assert preds.shape == (img_ax.size(0), 3), "Model output shape mismatch"
    assert not torch.isnan(preds).any(), "Model output contains NaNs"

    # 4. Loss Function Verification
    print("\n[4] Verifying Laplace Loss Function...")
    criterion = LaplaceLoss(cfg)

    # Prepare inputs for loss (Predictions vs Targets)
    alpha = preds[:, 0]
    sigma_base = preds[:, 1]
    sigma_growth = preds[:, 2]

    # Mock baseline data (usually retrieved from lookup dicts in Engine)
    # We create dummy baseline values for this unit test
    base_fvc = torch.full_like(target, 2000.0).to(device)
    base_week = torch.zeros_like(week).to(device)
    target = target.to(device)
    week = week.to(device)

    # Calculate Loss
    loss = criterion(alpha, sigma_base, sigma_growth, target, week, base_fvc, base_week)
    print(f"  Calculated Loss: {loss.item():.6f}")

    # Assertions for loss
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Full Engine Execution (Train & Inference)
    print("\n[5] Executing Engine (Training Loop & Submission)...")
    # Instantiate Engine
    engine = Engine(debug=True, epochs=1)

    # Run Training (Fit)
    print("  -> Starting training (1 epoch)...")
    engine.fit()

    # Verify Checkpoint Creation
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"  -> Checkpoint verified at: {best_model_path}")
    else:
        print(
            "  -> Note: No checkpoint saved (Validation score might not have improved in 1 epoch)."
        )

    # Run Inference (Generate Submission)
    print("  -> Generating submission...")
    engine.generate_submission()

    # Verify Submission File
    submission_path = cfg.SUBMISSION_PATH
    if os.path.exists(submission_path):
        print(f"  -> Submission file verified at: {submission_path}")

        # Load and validate content
        sub_df = pd.read_csv(submission_path)
        print("  -> Submission Head:")
        print(sub_df.head(3))

        expected_cols = ["Patient_Week", "FVC", "Confidence"]
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch. Got {sub_df.columns}"
        assert len(sub_df) > 0, "Submission file is empty"
        assert not sub_df.isnull().values.any(), "Submission contains null values"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n" + "=" * 40)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
