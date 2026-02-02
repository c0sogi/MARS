import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim
import warnings
import gc

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.model import SymmetryDifferenceSiameseNet
from library.engine import fit, predict_and_submit


def run_demo():
    print("Starting Library Usage Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Re-run setup to create the new working directory
    Config.setup()

    # Set reproducible seed
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Metric Verification
    # =========================================================================
    print("\n[2] Verifying Metric (probabilistic_f1)...")

    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(score_perfect, 1.0), f"Expected pF1=1.0, got {score_perfect}"

    # Case 2: Worst prediction
    y_pred_worst = np.array([0.0, 1.0, 0.0, 1.0])
    score_worst = probabilistic_f1(y_true, y_pred_worst)
    assert np.isclose(score_worst, 0.0), f"Expected pF1=0.0, got {score_worst}"

    print("    Metric logic verified.")

    # =========================================================================
    # 3. Data Pipeline Verification
    # =========================================================================
    print("\n[3] Verifying Data Pipeline...")

    # Force reload of data (load_cached_data=False) to test processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch a single batch to verify shapes
    try:
        inputs, labels = next(iter(train_loader))
        target_imgs, contra_imgs = inputs

        # Check Shapes
        # Expected: (Batch_Size, 3, 768, 768)
        assert target_imgs.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Target Image shape mismatch: {target_imgs.shape}"
        assert contra_imgs.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Contra Image shape mismatch: {contra_imgs.shape}"
        assert (
            labels.shape[0] == Config.BATCH_SIZE
        ), f"Label batch size mismatch: {labels.shape}"

        print("    Batch shapes verified successfully.")

        # Check content (Age/Implant channels)
        # Channel 1 is age, Channel 2 is implant. They should be constant spatially per image.
        # We check the first image in the batch.
        age_map = target_imgs[0, 1, :, :]
        implant_map = target_imgs[0, 2, :, :]

        assert torch.std(age_map) < 1e-6, "Age map should be spatially constant."
        assert (
            torch.std(implant_map) < 1e-6
        ), "Implant map should be spatially constant."

        print("    Metadata channels verified.")

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # =========================================================================
    # 4. Model Architecture Verification
    # =========================================================================
    print("\n[4] Verifying Model Architecture...")

    model = SymmetryDifferenceSiameseNet().to(device)

    # Move batch to device
    t_img = target_imgs.to(device)
    c_img = contra_imgs.to(device)

    # Forward Pass
    with torch.no_grad():
        logits = model((t_img, c_img))

    # Check Output Shape: (Batch_Size, 1)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("    Forward pass successful. Output shape correct.")

    # =========================================================================
    # 5. Training Loop Verification
    # =========================================================================
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Use the engine.fit function
    best_model_path = fit(
        train_loader, val_loader, epochs=Config.NUM_EPOCHS, device=device
    )

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"    Training finished. Model saved at {best_model_path}")

    # =========================================================================
    # 6. Inference & Submission Verification
    # =========================================================================
    print("\n[6] Running Inference and Generating Submission...")

    predict_and_submit(best_model_path, test_loader, device=device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = {"prediction_id", "cancer"}
    assert required_cols.issubset(
        df_sub.columns
    ), f"Missing columns in submission. Found: {df_sub.columns}"
    assert len(df_sub) > 0, "Submission file is empty."

    # Check probability range
    probs = df_sub["cancer"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("    Submission generated and verified successfully.")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
