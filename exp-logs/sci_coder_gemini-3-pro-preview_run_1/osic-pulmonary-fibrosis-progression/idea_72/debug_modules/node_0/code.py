import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders, TriSlabProcessor, LungDataset
from library.model import AASLNet
from library.train import Trainer
from library.utils import LaplaceLogLikelihoodLoss, score
from library.inference import predict

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting AASL-Net Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    # Modify Config attributes to run a tiny, fast experiment
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5  # Only use 5 patients for training/val
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead in small demo

    # Re-seed to ensure these changes take effect consistently
    seed_everything(Config.SEED)

    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Generate DataLoaders (this triggers caching of the debug subset)
    train_loader, val_loader = get_dataloaders(debug=True)

    # Fetch one batch to inspect structure
    try:
        batch = next(iter(train_loader))
        img_ax, img_cor, tabular, meta, target = batch

        print(f"    Batch loaded successfully.")
        print(f"    Axial Image Shape: {img_ax.shape} (Expected: B, 3, 224, 224)")
        print(f"    Coronal Image Shape: {img_cor.shape} (Expected: B, 3, 224, 224)")
        print(f"    Tabular Data Shape: {tabular.shape} (Expected: B, 4)")
        print(f"    Meta Data Shape: {meta.shape} (Expected: B, 2)")
        print(f"    Target Shape: {target.shape} (Expected: B, 1)")

        # Assertions
        assert img_ax.shape == (Config.BATCH_SIZE, 3, 224, 224), "Incorrect Axial shape"
        assert img_cor.shape == (
            Config.BATCH_SIZE,
            3,
            224,
            224,
        ), "Incorrect Coronal shape"
        assert tabular.shape == (Config.BATCH_SIZE, 4), "Incorrect Tabular shape"
        assert meta.shape == (Config.BATCH_SIZE, 2), "Incorrect Meta shape"

    except StopIteration:
        raise Exception("DataLoader is empty! Check dataset paths and debug split.")

    # ---------------------------------------------------------
    # 3. Model Architecture & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = AASLNet()
    model.to(Config.DEVICE)

    # Move batch to device
    img_ax = img_ax.to(Config.DEVICE)
    img_cor = img_cor.to(Config.DEVICE)
    tabular = tabular.to(Config.DEVICE)
    meta = meta.to(Config.DEVICE)
    target = target.to(Config.DEVICE)

    # Perform Forward Pass
    pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, meta)

    print(f"    Forward pass successful.")
    print(f"    Pred FVC Shape: {pred_fvc.shape}")
    print(f"    Pred Sigma Shape: {pred_sigma.shape}")

    # Assertions
    assert pred_fvc.shape == (Config.BATCH_SIZE,), "Output FVC shape mismatch"
    assert pred_sigma.shape == (Config.BATCH_SIZE,), "Output Sigma shape mismatch"
    assert not torch.isnan(pred_fvc).any(), "Model produced NaN in FVC"
    assert not torch.isnan(pred_sigma).any(), "Model produced NaN in Sigma"

    # ---------------------------------------------------------
    # 4. Loss and Metric Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss and Metric functions...")

    criterion = LaplaceLogLikelihoodLoss()

    # Calculate Loss
    loss = criterion(pred_fvc, pred_sigma, target)
    print(f"    Loss Value: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"

    # Calculate Metric (Score)
    # Score expects numpy/detached tensors
    metric_val = score(pred_fvc, pred_sigma, target)
    print(f"    Metric Score: {metric_val:.4f}")
    assert np.isfinite(metric_val), "Metric is not finite"

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Run Fit
    best_score = trainer.fit()

    print(f"    Training finished. Best Score: {best_score:.4f}")

    # Verify Model Checkpoint creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print(f"    Verified checkpoint exists at: {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 6. Inference Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n[6] Running Inference Pipeline (Limited Batches)...")

    # Run prediction for a limited number of batches to save time
    # This uses the model weights saved in step 5
    submission_df = predict(limit_batches=2)

    print(f"    Inference finished.")
    print(f"    Submission shape: {submission_df.shape}")
    print(f"    Columns: {list(submission_df.columns)}")

    # Assertions
    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "Patient_Week" in submission_df.columns
    assert "FVC" in submission_df.columns
    assert "Confidence" in submission_df.columns
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."

    print("\n=== Demonstration Complete: All Systems Operational ===")


if __name__ == "__main__":
    main()
