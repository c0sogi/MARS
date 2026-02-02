import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.trainer import fit
from library.inference import create_submission
from library.dataset import get_dataloaders
from library.model import ResNet34Model, set_seed


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    print("=== Starting Catheter Detection Pipeline ===")

    # 2. Training
    # Increased epochs to 10 to allow convergence with padded images.
    print("\n--- Step 1: Training Model ---")
    fit(epochs=10, batch_size=Config.BATCH_SIZE, debug=False)

    # 3. Validation and Metric Calculation
    print("\n--- Step 2: Validation & Metric Calculation ---")

    # Load validation dataloader
    loaders = get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False)
    val_loader = loaders["val"]

    # Load the best trained model
    device = Config.DEVICE
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = ResNet34Model(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    all_preds = []
    all_labels = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass (no gradient needed)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate AUC per column and average
    aucs = []
    for i in range(Config.NUM_CLASSES):
        # Only calculate AUC if the class has both 0 and 1 in the validation set
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)

    final_metric = np.mean(aucs) if aucs else 0.0

    # Print the required metric
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Step 3: Failure Analysis ---")

    # Calculate Mean Absolute Error (MAE) per sample
    # Shape: (N_samples,)
    mae_per_sample = np.mean(np.abs(all_preds - all_labels), axis=1)

    print("Correlation between Error Magnitude and Target Labels:")
    for i, col in enumerate(Config.TARGET_COLS):
        # Calculate point-biserial correlation
        if np.std(all_labels[:, i]) > 0:
            corr, _ = pearsonr(mae_per_sample, all_labels[:, i])
            print(f"  {col}: {corr:.4f}")

    print(
        "\nCorrelation between Error Magnitude and Input Features (Image Dimensions):"
    )
    # Extract image stats for a subset of the validation data to save time
    val_dataset = val_loader.dataset
    file_paths = val_dataset.file_paths

    # Sample up to 1000 images for analysis
    sample_size = min(1000, len(file_paths))
    indices = np.random.choice(len(file_paths), size=sample_size, replace=False)

    subset_errors = mae_per_sample[indices]
    widths = []
    heights = []
    aspect_ratios = []

    for idx in indices:
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, file_paths[idx])

        try:
            # Read image to get dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
                aspect_ratios.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Helper to print correlation
    def print_corr(name, feature_vals, errors):
        mask = ~np.isnan(feature_vals)
        if np.sum(mask) > 1 and np.std(feature_vals[mask]) > 0:
            corr, _ = pearsonr(errors[mask], feature_vals[mask])
            print(f"  {name}: {corr:.4f}")

    print_corr("Width", widths, subset_errors)
    print_corr("Height", heights, subset_errors)
    print_corr("Aspect Ratio", aspect_ratios, subset_errors)

    # 5. Submission
    print("\n--- Step 4: Generating Submission ---")
    threshold = 0.9396550795413087
    if final_metric > threshold:
        print(
            f"Validation metric {final_metric} > {threshold}. Generating submission..."
        )
        create_submission(batch_size=Config.BATCH_SIZE, debug=False)
    else:
        print(f"Validation metric {final_metric} <= {threshold}. Skipping submission.")

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
