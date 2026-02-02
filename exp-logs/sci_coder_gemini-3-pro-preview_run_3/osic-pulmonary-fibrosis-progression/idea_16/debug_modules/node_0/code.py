import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, metric_score
from library.data import get_dataloaders
from library.model import RaliNet
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    print("Starting RALI-Net Demo Execution...")

    # 1. Configuration & Setup
    # Override Config for a fast, isolated demo run
    Config.WORKING_DIR = "./working/demo_task_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_task_execution"
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = 224  # Slightly smaller for speed, though model uses pooling

    # Create directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading (Debug Mode)
    print("\n[1/5] Initializing DataLoaders (Debug Mode)...")
    # debug=True limits train to 50 rows, val to 20 rows
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=True,
        load_cached_data=True,
    )

    # Validate Data Shapes
    images, tabular, targets = next(iter(train_loader))
    print(
        f"  Batch Shapes -> Images: {images.shape}, Tabular: {tabular.shape}, Targets: {targets.shape}"
    )

    # Assertions
    assert images.ndim == 4, "Images should be 4D (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert tabular.shape[1] == 7, "Tabular data should have 7 features"
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"
    print("  DataLoaders validated.")

    # 3. Model Initialization & Forward Pass
    print("\n[2/5] Initializing RaliNet Model...")
    model = RaliNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)
    targets = targets.to(device)

    # Forward Pass
    preds = model(images, tabular)
    print(f"  Prediction Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        2,
    ), "Output shape must be (Batch, 2) [FVC, Sigma]"
    assert not torch.isnan(preds).any(), "Model output contains NaNs"
    print("  Model forward pass validated.")

    # 4. Loss & Metric Verification
    print("\n[3/5] Verifying Loss and Metric...")
    criterion = LaplaceLogLikelihoodLoss()

    # Calculate Loss
    loss = criterion(preds, targets)
    print(f"  Computed Loss: {loss.item():.4f}")
    assert loss.item() != 0, "Loss is zero"
    assert not torch.isnan(loss), "Loss is NaN"

    # Manual Metric Check
    # Scenario: True=2000, Pred=2100, Sigma=100
    # Delta = 100, Sigma_clipped = 100
    # Metric = - (sqrt(2)*100)/100 - ln(sqrt(2)*100)
    #        = -1.4142 - ln(141.42) = -1.4142 - 4.9517 = -6.3659
    y_true = np.array([2000])
    y_pred = np.array([2100])
    sigma_pred = np.array([100])
    score = metric_score(y_true, y_pred, sigma_pred)
    print(f"  Manual Metric Check (Expected ~ -6.3659): {score:.4f}")
    assert -6.37 < score < -6.36, "Metric calculation logic mismatch"
    print("  Loss and Metric validated.")

    # 5. Training Loop Demonstration
    print("\n[4/5] Executing Training Loop (1 Epoch)...")

    # Setup Optimizer (simplified for demo)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"  Epoch 1 Train Loss: {train_loss:.4f}")

    # Evaluate
    val_loss, val_metric = evaluate(model, val_loader, criterion, device, stats)
    print(f"  Epoch 1 Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")

    # Save Checkpoint
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Checkpoint file was not created"
    print("  Training loop and checkpointing validated.")

    # 6. Submission Generation
    print("\n[5/5] Generating Submission...")
    generate_submission(model, test_loader, device, stats)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file missing"
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)

    print(f"  Submission Head:\n{sub_df.head(3)}")

    # Check Columns
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission"

    # Check Row Count (Test set has 18 patients, submission expands to all weeks)
    # Just check it's not empty
    assert len(sub_df) > 0, "Submission file is empty"

    # Check Values
    assert sub_df["Confidence"].min() >= 70, "Confidence values should be clipped at 70"

    print("  Submission generation validated.")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
