import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.model import DeformableSiameseModel
from library.train import train_one_epoch, validate, generate_submission


def run_demo():
    print("Starting Breast Cancer Detection Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Set a specific working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config class attributes directly
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small sample size for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = (256, 256)  # Smaller image size for faster processing

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] initializing DataLoaders...")

    # We disable loading cached data to ensure the pipeline runs from scratch
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    target_img, contra_img, labels = batch

    print(
        f"    Batch Shapes -> Target: {target_img.shape}, Contra: {contra_img.shape}, Labels: {labels.shape}"
    )

    # Assertions
    # Expected shape: (B, 3, H, W) because of [Image, Age, Implant] channels
    assert target_img.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Unexpected target image shape: {target_img.shape}"
    assert contra_img.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Unexpected contra image shape: {contra_img.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected labels shape: {labels.shape}"

    # -------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Instantiating Model...")

    # Use pretrained=False to avoid downloading weights during demo
    model = DeformableSiameseModel(pretrained=False)
    model = model.to(device)

    print("    Performing dummy forward pass...")
    target_img = target_img.to(device)
    contra_img = contra_img.to(device)

    with torch.no_grad():
        logits = model(target_img, contra_img)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    # -------------------------------------------------------------------------
    # 4. Metric Verification (Probabilistic F1)
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Metric Calculation...")

    # Test Case 1: Perfect Prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)
    print(f"    pF1 (Perfect): {score_perfect:.4f}")
    assert np.isclose(score_perfect, 1.0), "pF1 should be 1.0 for perfect predictions"

    # Test Case 2: All Zeros (Recall should be 0)
    y_pred_zeros = np.array([0.0, 0.0, 0.0, 0.0])
    score_zeros = probabilistic_f1(y_true, y_pred_zeros)
    print(f"    pF1 (All Zeros): {score_zeros:.4f}")
    assert np.isclose(score_zeros, 0.0), "pF1 should be 0.0 for zero predictions"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Step...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch (on the tiny debug dataset)
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Validation Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Validation Step...")

    val_loss, val_pf1 = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val pF1:  {val_pf1:.4f}")

    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0.0 <= val_pf1 <= 1.0, "pF1 score must be between 0 and 1"

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission...")

    # Save the current model state (simulating a 'best model')
    model_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    torch.save(model.state_dict(), model_path)

    # Generate submission using the test loader
    generate_submission(model_path, test_loader, device)

    # Verify output file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file created with {len(df_sub)} rows.")
        print(f"    First few rows:\n{df_sub.head()}")

        # Check columns
        assert "prediction_id" in df_sub.columns, "Missing prediction_id column"
        assert "cancer" in df_sub.columns, "Missing cancer column"

        # Check value range
        assert (
            df_sub["cancer"].min() >= 0.0 and df_sub["cancer"].max() <= 1.0
        ), "Probabilities must be between 0 and 1"
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
