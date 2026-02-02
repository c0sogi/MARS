import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ==========================================
# 1. CONFIGURATION & PATCHING
# ==========================================
# Import config and train modules
import library.config
import library.train

# Monkey-patch configuration for Fast Baseline execution
# Reducing epochs and patience to ensure completion within 2 hours
library.config.MAX_EPOCHS = 25
library.train.MAX_EPOCHS = 25
library.config.PATIENCE = 5
library.train.PATIENCE = 5

# Import necessary components after patching
from library.config import DEVICE, SUBMISSION_PATH, CACHE_PATH
from library.utils import seed_everything
from library.data_loader import load_data
from library.train import run_fold, validate, predict_test
from library.model import DN_WBN


def main():
    # Set seeds for reproducibility
    seed_everything(42)

    print("========================================")
    print("      DN-WBN Fast Baseline Pipeline     ")
    print("========================================")

    # ==========================================
    # 2. DATA LOADING
    # ==========================================
    print("\n[1/5] Loading Data...")
    # load_data returns loaders corresponding to metadata/train.csv and metadata/val.csv
    train_loader, val_loader, test_loader = load_data(load_cached_data=True)

    # ==========================================
    # 3. TRAINING (Fixed Split)
    # ==========================================
    print("\n[2/5] Training Model (Fold 0 / Fixed Split)...")
    # We use run_fold(0) to train on the specific train_loader provided
    # This saves the model to 'dn_wbn_fold_0.pth' and returns the best state
    model = run_fold(0, train_loader, val_loader, DEVICE)

    # ==========================================
    # 4. VALIDATION EVALUATION
    # ==========================================
    print("\n[3/5] Evaluating on Hold-out Validation Set...")
    # Criterion for loss calculation (BCE)
    criterion = nn.BCELoss()

    # Calculate final metric on the hold-out set
    val_loss, val_metric = validate(model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric:.15f}")

    # ==========================================
    # 5. FAILURE ANALYSIS
    # ==========================================
    print("\n[4/5] Performing Failure Analysis...")
    model.eval()

    all_preds = []
    all_labels = []
    all_angles = []
    all_img_means = []
    all_img_stds = []

    # Efficient inference without gradients
    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(DEVICE)
            angles = angles.to(DEVICE)

            # Forward pass
            outputs = model(images, angles)

            # Collect data
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_angles.extend(angles.cpu().numpy())

            # Compute simple image stats for analysis (on CPU numpy arrays)
            # images shape: (B, 3, 75, 75)
            # We average across the spatial dimensions and channels for a global brightness metric
            imgs_np = images.cpu().numpy()
            # Calculate mean and std per image in the batch
            batch_means = np.mean(imgs_np, axis=(1, 2, 3))
            batch_stds = np.std(imgs_np, axis=(1, 2, 3))

            all_img_means.extend(batch_means)
            all_img_stds.extend(batch_stds)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_angles = np.array(all_angles)
    all_img_means = np.array(all_img_means)
    all_img_stds = np.array(all_img_stds)

    # Calculate Error Magnitude (Absolute Difference)
    errors = np.abs(all_labels - all_preds)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error_magnitude": errors,
            "incidence_angle": all_angles,
            "image_mean_intensity": all_img_means,
            "image_std_intensity": all_img_stds,
        }
    )

    # Drop any NaNs (though angles should be imputed by loader)
    df_analysis = df_analysis.dropna()

    # Calculate correlations
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 6. SUBMISSION GENERATION
    # ==========================================
    print("\n[5/5] Checking Submission Criteria...")
    THRESHOLD = 0.15744295919935183

    if val_metric < THRESHOLD:
        print(
            f"Metric ({val_metric:.6f}) < Threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Generate predictions for test set
        ids, preds = predict_test(model, test_loader, DEVICE)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": ids, "is_iceberg": preds})

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {SUBMISSION_PATH}")

    else:
        print(
            f"Metric ({val_metric:.6f}) >= Threshold ({THRESHOLD:.6f}). Submission skipped."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
