import os
import torch
import numpy as np
import pandas as pd
from scipy import stats

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.model import ModifiedDenseNet
from library.dataset import get_dataloaders
from library.train import run_training
from library.predict import predict_submission


def main():
    # 1. Setup and Configuration
    # Set fixed random seed for reproducibility
    set_seed(Config.SEED)

    # Adjust training epochs to ensure the "fast baseline" completes within the 2-hour limit.
    # The dataset has ~140k training images. With batch size 32, one epoch is ~4360 steps.
    # On an A100, 12 epochs should comfortably fit within the time budget while allowing convergence.
    Config.EPOCHS = 12

    print(f"Configuration set: Epochs={Config.EPOCHS}, Device={Config.DEVICE}")

    # 2. Training
    print("\n" + "=" * 30)
    print("Phase 1: Model Training")
    print("=" * 30)
    # Run training (handles training loop, validation monitoring, and saving best model)
    run_training(debug=False)

    # 3. Validation and Failure Analysis
    print("\n" + "=" * 30)
    print("Phase 2: Validation & Failure Analysis")
    print("=" * 30)

    # Load the best model saved during training
    device = torch.device(Config.DEVICE)
    model = ModifiedDenseNet(pretrained=False)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    print(f"Loading best model from {Config.MODEL_PATH}...")
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get validation dataloader
    # debug=False ensures we use the full validation set
    _, val_loader, _ = get_dataloaders(debug=False)

    all_preds = []
    all_targets = []

    # Lists to store image meta-features for failure analysis
    # We will correlate error magnitude with image brightness and contrast
    feat_brightness = []
    feat_contrast = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Store predictions and targets
            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

            # Calculate simple image statistics on the normalized tensors
            # Brightness: Mean intensity across channels and spatial dims
            # Contrast: Standard deviation across channels and spatial dims
            # shape: (B, C, H, W) -> reduce over (1, 2, 3)
            batch_mean = torch.mean(images, dim=(1, 2, 3)).cpu().numpy()
            batch_std = torch.std(images, dim=(1, 2, 3)).cpu().numpy()

            feat_brightness.extend(batch_mean)
            feat_contrast.extend(batch_std)

    # Convert collected lists to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    brightness = np.array(feat_brightness)
    contrast = np.array(feat_contrast)

    # Calculate and print the required metric
    val_auc = calculate_roc_auc(y_true, y_pred)
    print(f"Final Validation Metric: {val_auc}")

    # Perform Failure Analysis
    # Calculate error magnitude: |y_true - y_pred|
    errors = np.abs(y_true - y_pred)

    # Calculate Pearson correlation between error and image features
    corr_bright, _ = stats.pearsonr(errors, brightness)
    corr_contrast, _ = stats.pearsonr(errors, contrast)

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(f"  Brightness: {corr_bright}")
    print(f"  Contrast:   {corr_contrast}")

    # 4. Submission Generation
    print("\n" + "=" * 30)
    print("Phase 3: Submission Generation")
    print("=" * 30)

    # Threshold defined in the task
    THRESHOLD = 0.9734124924386656

    if val_auc > THRESHOLD:
        print(f"Validation metric {val_auc} exceeds threshold {THRESHOLD}.")
        print("Generating predictions for test set...")
        predict_submission(debug=False)
    else:
        print(f"Validation metric {val_auc} does not exceed threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
