import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import quadratic_weighted_kappa, AverageMeter
from library.data import load_metadata, get_dataloaders, RetinopathyDataset
from library.model import DRModel
from library.engine import train_one_epoch, valid_one_epoch, EarlyStopping


def run_demo():
    # 1. Setup and Reproducibility
    print(">>> Setting up environment and seeds...")
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration
    # We use a smaller backbone and disable pretraining to avoid downloads/timeouts
    print(">>> Overriding Config for fast demonstration...")
    Config.BACKBONE = "resnet18"
    Config.PRETRAINED = False
    Config.BATCH_SIZE = 8
    Config.PHASE_1_RES = 256  # Smaller resolution for speed
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading and Subsampling
    print("\n>>> Loading and subsampling metadata...")
    train_df, val_df, test_df = load_metadata()

    # Use a tiny subset for demonstration (e.g., 32 samples each)
    # This ensures the code runs in seconds rather than hours
    demo_size = 32
    train_df = train_df.iloc[:demo_size].reset_index(drop=True)
    val_df = val_df.iloc[:demo_size].reset_index(drop=True)
    test_df = test_df.iloc[:demo_size].reset_index(drop=True)

    print(f"Train subset shape: {train_df.shape}")
    print(f"Val subset shape: {val_df.shape}")
    print(f"Test subset shape: {test_df.shape}")

    # 3. DataLoader Instantiation
    print("\n>>> Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df,
        val_df,
        test_df,
        image_size=Config.PHASE_1_RES,
        batch_size=Config.BATCH_SIZE,
    )

    # Verify DataLoader output
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.PHASE_1_RES,
        Config.PHASE_1_RES,
    ), "Incorrect image batch shape"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect target batch shape"

    # 4. Model Initialization
    print("\n>>> Initializing Model (ResNet18 backbone)...")
    model = DRModel(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        gem_p=Config.GEM_P,
    )
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.PHASE_1_RES, Config.PHASE_1_RES).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        1,
    ), "Model output shape should be (Batch, 1) for regression"

    # 5. Metric Verification
    print("\n>>> Verifying Quadratic Weighted Kappa Metric...")
    # Case 1: Perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0.1, 1.1, 1.9, 3.0, 4.0])  # Floats close to integers
    score = quadratic_weighted_kappa(y_true, y_pred)
    print(f"Perfect Agreement Score (approx): {score}")
    assert np.isclose(score, 1.0), "QWK should be 1.0 for perfect agreement"

    # Case 2: Random/Poor agreement
    y_pred_bad = np.array([4, 3, 2, 1, 0])
    score_bad = quadratic_weighted_kappa(y_true, y_pred_bad)
    print(f"Poor Agreement Score: {score_bad}")
    assert score_bad < 1.0, "QWK should be < 1.0 for poor agreement"

    # 6. Training Loop Simulation
    print("\n>>> Simulating Training Epoch...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"
    assert train_loss >= 0, "Training loss should be non-negative (MSE)"

    # 7. Validation Loop Simulation
    print("\n>>> Simulating Validation Epoch...")
    val_loss, val_qwk, val_preds, val_targets = valid_one_epoch(
        model, val_loader, device
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation QWK: {val_qwk:.4f}")

    assert len(val_preds) == len(
        val_df
    ), "Number of predictions does not match validation set size"
    assert (
        val_preds.shape == val_targets.shape
    ), "Predictions and targets shape mismatch"

    # 8. Early Stopping Check
    print("\n>>> Checking Early Stopping Logic...")
    early_stopper = EarlyStopping(
        patience=2, mode="max", save_path="./working/temp_best_model.pth"
    )

    # Simulate improvement
    early_stopper(0.5, model)
    assert early_stopper.best_score == 0.5
    assert early_stopper.counter == 0
    assert os.path.exists("./working/temp_best_model.pth"), "Model checkpoint not saved"

    # Simulate no improvement
    early_stopper(0.4, model)
    assert early_stopper.counter == 1

    # 9. Test/Inference Simulation
    print("\n>>> Simulating Inference on Test Set...")
    model.eval()
    test_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_preds.extend(outputs.view(-1).cpu().numpy())

    test_preds = np.array(test_preds)

    # Convert regression outputs to ordinal labels (0-4)
    final_predictions = np.rint(test_preds).clip(0, 4).astype(int)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"id_code": test_df["id_code"], "diagnosis": final_predictions}
    )

    print("Sample Submission:")
    print(submission.head())

    assert len(submission) == len(test_df), "Submission length mismatch"
    assert (
        submission["diagnosis"].between(0, 4).all()
    ), "Predictions out of range [0, 4]"

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
