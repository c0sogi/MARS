import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import NBCSLN
from library.utils import LaplaceLogLikelihoodLoss, calculate_metric
from library.engine import train_fn, eval_fn, inference_fn


def run_demo():
    print("=" * 40)
    print("LUNG FUNCTION DECLINE PREDICTION DEMO")
    print("=" * 40)

    # 1. Configuration & Setup
    print("\n[1] Setting up configuration...")
    seed_everything(42)

    # Override Config for fast demonstration execution
    # We use the DEBUG flag to limit the dataset size significantly
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Data Loading
    print("\n[2] Loading data and generating image cache...")
    # get_dataloaders handles reading CSVs, processing DICOMs to .npy, and creating loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Data Loading Logic
    print("    Verifying DataLoader batch structure...")
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    required_keys = [
        "img_axial",
        "img_coronal",
        "tabular",
        "meta",
        "target",
        "patient_week",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Tensor Shapes
    # Images: (B, 3, 224, 224)
    assert batch["img_axial"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert batch["img_coronal"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    # Tabular: (B, 7)
    assert batch["tabular"].shape == (Config.BATCH_SIZE, 7)
    # Meta: (B, 2) -> [Baseline_FVC, Week_Diff]
    assert batch["meta"].shape == (Config.BATCH_SIZE, 2)
    # Target: (B, 1)
    assert batch["target"].shape == (Config.BATCH_SIZE, 1)

    print("    DataLoader verification passed.")

    # 3. Model Initialization
    print("\n[3] Initializing NBCSLN model...")
    model = NBCSLN()
    model.to(device)

    # Verify Model Forward Pass
    print("    Verifying model forward pass...")
    img_ax = batch["img_axial"].to(device)
    img_cor = batch["img_coronal"].to(device)
    tab = batch["tabular"].to(device)

    with torch.no_grad():
        # Model expects: axial_img, coronal_img, tabular_features
        output = model(img_ax, img_cor, tab)

    # Output should be (B, 3) -> [alpha, sigma_base, sigma_growth]
    assert output.shape == (
        Config.BATCH_SIZE,
        3,
    ), f"Model output shape mismatch: {output.shape}"
    print("    Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n[4] Running training step (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = LaplaceLogLikelihoodLoss()

    # Run one epoch of training using the engine's train_fn
    avg_loss = train_fn(train_loader, model, optimizer, device, loss_fn)
    print(f"    Training Epoch 1 Loss: {avg_loss:.4f}")

    # Sanity check on loss
    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss != 0, "Training loss is zero (unexpected)"

    # 5. Validation Loop Demonstration
    print("\n[5] Running validation step...")
    # Run validation using the engine's eval_fn
    val_metric = eval_fn(val_loader, model, device)
    print(f"    Validation Metric: {val_metric:.4f}")

    # Sanity check on metric
    assert not np.isnan(val_metric), "Validation metric is NaN"

    # 6. Inference Demonstration
    print("\n[6] Running inference on test set...")
    # Generate predictions using the engine's inference_fn
    submission_df = inference_fn(test_loader, model, device)

    # Verify Submission Format
    print("    Verifying submission format...")
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in expected_cols:
        assert col in submission_df.columns, f"Submission missing column: {col}"

    # Check if we have rows (DEBUG mode limits test set to 20 rows)
    num_rows = len(submission_df)
    print(f"    Generated {num_rows} predictions.")
    assert num_rows > 0, "Submission DataFrame is empty"

    # 7. Saving Submission
    save_path = Config.SUBMISSION_FILE
    # Ensure column order
    submission_df = submission_df[expected_cols]
    submission_df.to_csv(save_path, index=False)
    print(f"\n[7] Submission saved to {save_path}")
    print(submission_df.head())

    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
