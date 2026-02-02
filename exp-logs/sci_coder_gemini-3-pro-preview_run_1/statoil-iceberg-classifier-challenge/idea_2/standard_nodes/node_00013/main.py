import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import sys
import os

# Import from provided library files
from library.config import SEED, DEVICE, SUBMISSION_PATH
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.trainer import train_model, predict_and_submit


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Training
    # Increased epochs to allow convergence with label smoothing and stronger augmentation
    print("Starting training...")
    model = train_model(train_loader, val_loader, num_epochs=25)

    # 4. Validation & Metric Calculation
    print("Running validation inference with TTA...")
    model.eval()

    val_preds = []
    val_targets = []
    val_angles = []
    val_img_means = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(DEVICE)
            angles_gpu = angles.to(DEVICE)

            # TTA: Original
            out1 = model(images, angles_gpu)

            # TTA: Horizontal Flip
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles_gpu)

            # TTA: Vertical Flip
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles_gpu)

            # Average
            outputs = (out1 + out2 + out3) / 3.0

            # Store results
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(labels.numpy().flatten())
            val_angles.extend(angles.numpy().flatten())

            # Calculate mean intensity for failure analysis
            # images shape: (B, 3, H, W) -> flatten spatial dims -> mean
            batch_means = images.view(images.size(0), -1).mean(dim=1).cpu().numpy()
            val_img_means.extend(batch_means)

    # Convert to numpy arrays
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_angles = np.array(val_angles)
    val_img_means = np.array(val_img_means)

    # Compute Metric (Log Loss)
    # Clip predictions slightly to avoid log(0) though BCELoss handles it, sklearn might be sensitive
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    metric = log_loss(val_targets, val_preds_clipped)

    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (absolute difference)
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "inc_angle": val_angles, "image_mean": val_img_means}
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.21099163245555455
    if metric < THRESHOLD:
        print(
            f"\nValidation metric ({metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, test_ids, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
