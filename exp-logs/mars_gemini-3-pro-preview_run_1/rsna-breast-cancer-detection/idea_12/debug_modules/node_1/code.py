import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.model import FlowAlignedSiameseNet
from library.engine import fit, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Demonstration ====")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for rapid demonstration
    Config.DEBUG = True
    Config.DEBUG_DATA_SIZE = 32  # Small subset for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for demo speed.")

    # ---------------------------------------------------------
    # 2. Verify Metric Logic (Probabilistic F1)
    # ---------------------------------------------------------
    print("\n[2] Verifying Probabilistic F1 Metric...")

    # Manual Calculation Case
    # y_true: [1, 0]
    # y_pred: [0.8, 0.2]
    # pTP = (1*0.8) + (0*0.2) = 0.8
    # pFP = (0*0.8) + (1*0.2) = 0.2
    # TotalPos = 1 + 0 = 1
    # pPrecision = 0.8 / (0.8 + 0.2) = 0.8
    # pRecall = 0.8 / 1.0 = 0.8
    # pF1 = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8

    y_true = np.array([1, 0])
    y_pred = np.array([0.8, 0.2])
    score = probabilistic_f1(y_true, y_pred)

    print(f"Calculated pF1: {score:.4f}")

    # Assertion with small epsilon tolerance
    assert abs(score - 0.8) < 1e-6, f"pF1 calculation failed. Expected 0.8, got {score}"
    print("Metric verification passed.")

    # ---------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n[3] Initializing Data Pipeline...")

    # Generate DataLoaders (this handles metadata processing and scaling internally)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch to inspect
    batch = next(iter(train_loader))

    target = batch["target"]
    contra = batch["contra"]
    label = batch["label"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Target Tensor Shape: {target.shape}")  # Expected: (B, 3, H, W)
    print(f"Contra Tensor Shape: {contra.shape}")  # Expected: (B, 3, H, W)
    print(f"Label Tensor Shape:  {label.shape}")  # Expected: (B)

    # Assertions
    assert target.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Unexpected target shape: {target.shape}"
    assert contra.shape == target.shape, "Target and Contralateral shapes mismatch"
    assert label.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    # Verify Channel Stacking (Image, Age, Implant)
    # Age map should be constant per image
    age_channel = target[0, 1, :, :]
    assert torch.min(age_channel) == torch.max(
        age_channel
    ), "Age channel is not spatially constant"
    print("Data Pipeline verification passed.")

    # ---------------------------------------------------------
    # 4. Model Demonstration
    # ---------------------------------------------------------
    print("\n[4] Initializing Model...")

    model = FlowAlignedSiameseNet()
    model.to(device)

    print(f"Model backbone: {Config.BACKBONE}")

    # Forward pass verification
    print("Running forward pass on sample batch...")
    with torch.no_grad():
        target_dev = target.to(device)
        contra_dev = contra.to(device)
        logits = model(target_dev, contra_dev)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("Model forward pass verification passed.")

    # ---------------------------------------------------------
    # 5. Training Loop (Engine) Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (Fit)...")

    # Run fit (trains for 1 epoch as per modified Config)
    # This tests train_one_epoch, validate, saving checkpoints, etc.
    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    print("Training loop completed.")

    # Check if checkpoint exists
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint verified at: {best_model_path}")
    else:
        # It's possible validation didn't improve if initialized weights were lucky/unlucky,
        # but fit() should save at least once if pF1 > -1.
        # However, with random init, pF1 might be 0.
        print(
            "Note: Checkpoint might not exist if metric did not improve (unlikely with init -1)."
        )

    # ---------------------------------------------------------
    # 6. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[6] Running Inference...")

    submission_df = inference(trained_model, test_loader, device)

    print("Inference completed.")
    print(submission_df.head())

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file saved at: {Config.SUBMISSION_PATH}")

        # Verify Schema
        df_check = pd.read_csv(Config.SUBMISSION_PATH)
        required_cols = {"prediction_id", "cancer"}
        assert required_cols.issubset(
            df_check.columns
        ), f"Submission missing columns. Found: {df_check.columns}"
        assert len(df_check) > 0, "Submission file is empty"

        # Verify Probabilities
        if not df_check.empty:
            probs = df_check["cancer"]
            assert (
                probs.min() >= 0.0 and probs.max() <= 1.0
            ), "Predictions out of probability range [0, 1]"

        print("Submission file verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
