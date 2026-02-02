import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import sys
import os

# Import from provided library files
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import DRPPN, train_model, predict_and_submit


def run():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using batch_size=32 as per constraints and memory safety
    print("Loading data...")
    train_loader, val_loader, test_loader, ids_test = get_dataloaders(
        batch_size=32, load_cached_data=True
    )

    # 3. Model Initialization
    model = DRPPN()

    # 4. Training
    # Limiting to 30 epochs for a fast baseline execution while respecting the low LR strategy
    print("Starting training...")
    trained_model = train_model(
        model, train_loader, val_loader, epochs=30, patience=10, device=device
    )

    # 5. Validation & Metric Calculation
    print("Performing validation inference...")
    trained_model.eval()
    y_true = []
    y_pred = []

    # Lists for failure analysis
    angles_list = []
    img_means_list = []

    with torch.no_grad():
        for inputs, angles, labels in val_loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = trained_model(inputs, angles)
            probs = torch.sigmoid(logits)

            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(probs.cpu().numpy().flatten())

            # Store features for failure analysis
            angles_list.extend(angles.cpu().numpy())
            # Calculate mean intensity of the image (across channels and pixels) for analysis
            img_means_list.extend(inputs.cpu().numpy().mean(axis=(1, 2, 3)))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to avoid log(0) errors, though sigmoid handles this mostly
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(y_true - y_pred)

    # Correlation with Incidence Angle
    corr_angle, _ = pearsonr(errors, angles_list)
    print(f"Correlation between Error and Incidence Angle: {corr_angle:.4f}")

    # Correlation with Image Intensity
    corr_intensity, _ = pearsonr(errors, img_means_list)
    print(f"Correlation between Error and Image Intensity: {corr_intensity:.4f}")

    # 7. Conditional Submission
    threshold = 0.20320119103524176
    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({threshold}). Generating submission..."
        )
        predict_and_submit(
            trained_model,
            test_loader,
            ids_test,
            output_dir="./submission",
            device=device,
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
