import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ShallowCNN
from library.engine import train_one_epoch, evaluate, EarlyStopping, predict_and_submit


def perform_failure_analysis(model, loader, device):
    """
    Performs inference on validation set to compute metrics and analyze failure modes.
    Returns the AUC and prints correlation analysis.
    """
    model.eval()
    all_targets = []
    all_preds = []

    # Lists to store meta-features for failure analysis
    brightness_list = []
    contrast_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets)

            # Calculate image stats for failure analysis
            # images shape: (B, 3, 48, 48)
            # Compute mean/std per image across C, H, W
            # We can compute on GPU then move to CPU

            # Brightness: Mean pixel value
            b_batch = torch.mean(images, dim=(1, 2, 3)).cpu().numpy()
            brightness_list.extend(b_batch)

            # Contrast: Std of pixel values
            c_batch = torch.std(images, dim=(1, 2, 3)).cpu().numpy()
            contrast_list.extend(c_batch)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    brightness = np.array(brightness_list)
    contrast = np.array(contrast_list)

    # 1. Calculate Final Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 2. Failure Analysis
    # Error magnitude = |Target - Prediction|
    # For binary classification:
    # If Target=1, Error = 1 - Prob
    # If Target=0, Error = Prob - 0 = Prob
    # This simplifies to abs(Target - Prob)
    errors = np.abs(all_targets - all_preds)

    print("\nFailure Analysis (Correlation with Error Magnitude):")

    # Correlation with Brightness
    if len(np.unique(brightness)) > 1:
        corr_b, p_b = stats.pearsonr(errors, brightness)
        print(f"  Brightness vs Error: Correlation={corr_b:.6f} (p={p_b:.6e})")
    else:
        print("  Brightness vs Error: N/A (Constant values)")

    # Correlation with Contrast
    if len(np.unique(contrast)) > 1:
        corr_c, p_c = stats.pearsonr(errors, contrast)
        print(f"  Contrast vs Error:   Correlation={corr_c:.6f} (p={p_c:.6e})")
    else:
        print("  Contrast vs Error:   N/A (Constant values)")

    return val_auc


def main():
    # 1. Configuration Overrides for Fast Baseline
    # Increase batch size to leverage A100
    Config.BATCH_SIZE = 256
    # Limit epochs for speed (fast baseline)
    Config.NUM_EPOCHS = 5
    # Ensure reproducibility
    Config.set_seed(Config.SEED)
    # Setup directories
    Config.setup_directories()

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use the full dataset but fewer epochs. The A100 is fast enough.
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_sample_size=None,  # Use full dataset
    )

    # 3. Model Initialization
    model = ShallowCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=Config.MODEL_SAVE_PATH,
    )

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        early_stopping(val_auc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Validation & Failure Analysis
    print("\nLoading best model for analysis...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    print("Performing validation and failure analysis...")
    perform_failure_analysis(model, val_loader, device)

    # 6. Submission
    print("\nGenerating submission for test set...")
    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
