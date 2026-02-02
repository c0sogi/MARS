import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, score_function
from library.loss import MetricAlignedLLLoss
from library.data import get_dataloaders
from library.model import MACRNet
from library.train import Trainer
from library.evaluate import predict_test_set


def run_demo():
    print("=== Starting Lung Function Prediction Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for a fast demo run
    Config.IDEA_ID = "demo_execution"
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Update paths based on new IDEA_ID
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_ID)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated for demo execution.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    try:
        train_loader, val_loader, test_loader, stats = get_dataloaders(
            debug=Config.DEBUG
        )

        print(f"Stats computed: {stats}")

        # Fetch one batch
        imgs, tabs, targets = next(iter(train_loader))

        print(
            f"Batch Shapes -> Images: {imgs.shape}, Tabular: {tabs.shape}, Targets: {targets.shape}"
        )

        # Assertions
        assert imgs.ndim == 4, "Images should be 4D (B, C, H, W)"
        assert imgs.shape[1] == 3, "Images should have 3 channels (slices)"
        assert tabs.shape[1] == 5, "Tabular data should have 5 features"
        assert targets.ndim == 1, "Targets should be 1D"

        print("Data Pipeline verification passed.")

    except Exception as e:
        print(f"Data Pipeline failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 3. Model and Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = MACRNet().to(device)
    criterion = MetricAlignedLLLoss()

    # Create dummy input
    dummy_img = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_tab = torch.randn(4, 5).to(device)
    dummy_target = torch.randn(4, 1).to(device)

    # Forward pass
    preds = model(dummy_img, dummy_tab)
    print(f"Model Output Shape: {preds.shape}")

    assert preds.shape == (4, 2), "Model output should be (Batch, 2) [mu, raw_sigma]"

    # Loss calculation
    loss = criterion(preds, dummy_target)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.ndim == 0, "Loss should be a scalar"

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric Verification (Laplace Log Likelihood)
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Metric Logic...")

    # Case 1: Perfect prediction with high confidence
    # Sigma clipped to 70. Delta = 0.
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([50.0])  # Should be clipped to 70

    score = score_function(y_true, y_pred, sigma)
    print(f"Metric Score (Perfect Pred, Low Sigma): {score:.4f}")

    expected_score = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        score, expected_score, atol=1e-3
    ), f"Metric mismatch. Got {score}, expected {expected_score}"

    # Case 2: Large Error, clipped at 1000
    y_true_2 = np.array([2000.0])
    y_pred_2 = np.array([4000.0])  # Error 2000 -> Clipped to 1000
    sigma_2 = np.array([100.0])

    score_2 = score_function(y_true_2, y_pred_2, sigma_2)
    print(f"Metric Score (Large Error): {score_2:.4f}")

    # Expected: - (sqrt(2)*1000)/100 - ln(sqrt(2)*100)
    # = -14.142 - ln(141.42) = -14.142 - 4.951 = -19.093
    expected_score_2 = -(np.sqrt(2) * 1000) / 100 - np.log(np.sqrt(2) * 100)
    assert np.isclose(
        score_2, expected_score_2, atol=1e-3
    ), f"Metric mismatch. Got {score_2}, expected {expected_score_2}"

    print("Metric logic verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    trainer = Trainer(debug=True)

    # Run fit
    trainer.fit(epochs=Config.EPOCHS)

    # Verify checkpoint creation
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Checkpoint successfully created at {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Training finished but best_model.pth was not found.")

    print("Training loop execution passed.")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    predict_test_set(debug=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")

        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission Head:")
        print(sub_df.head())

        assert "Patient_Week" in sub_df.columns
        assert "FVC" in sub_df.columns
        assert "Confidence" in sub_df.columns
        assert len(sub_df) > 0

        # Check confidence clipping in output
        min_conf = sub_df["Confidence"].min()
        print(f"Minimum Predicted Confidence: {min_conf}")
        assert (
            min_conf >= 70.0
        ), "Confidence values in submission are below the clip threshold of 70."

    else:
        raise FileNotFoundError("Inference finished but submission.csv was not found.")

    print("Inference execution passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
