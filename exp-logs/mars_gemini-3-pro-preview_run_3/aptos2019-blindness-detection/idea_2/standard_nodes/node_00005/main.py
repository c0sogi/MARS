import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import cv2
from scipy.stats import spearmanr

# Append current directory to system path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders
from library.model import EfficientNetRegressor
from library.train import train_one_epoch, validate_one_epoch, predict_test


def get_raw_predictions(model, loader, device):
    """
    Runs inference and returns raw continuous predictions and ground truth labels.
    Used for failure analysis and final metric calculation.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Labels might be needed for alignment, though we assume loader order is preserved
            all_labels.append(labels.numpy())

            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate and flatten
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    return all_preds, all_labels


def perform_failure_analysis(df_val, raw_preds, labels):
    """
    Calculates correlation between prediction error and image meta-features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude (using continuous predictions for granularity)
    # Ensure labels are float for subtraction
    errors = np.abs(labels - raw_preds)

    # Extract Meta-Features
    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    print("Extracting meta-features from validation set...")
    for idx, row in df_val.iterrows():
        # Construct full path
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image to get stats
        try:
            img = cv2.imread(file_path)
            if img is None:
                # Fallback for missing images
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
                intensities.append(0)
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)

            # Mean intensity
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            intensities.append(img_rgb.mean() / 255.0)

        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            intensities.append(0)

    # Compute Spearman Correlations
    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "Mean Intensity": intensities,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in features.items():
        if len(values) != len(errors):
            print(f"  {name}: Skipped (Length mismatch)")
            continue

        # Spearman correlation is appropriate for non-linear monotonic relationships
        corr, _ = spearmanr(errors, values)
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Ensure submission directory exists
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # We use the full dataset as it's small (~3k images) and fits the 'fast' constraint
    # while maximizing performance for the high threshold.
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    print("Initializing model...")
    model = EfficientNetRegressor()
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=Config.USE_AMP)

    # 4. Training Loop
    epochs = Config.EPOCHS
    best_qwk = -np.inf
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, loss_fn
        )
        val_loss, val_qwk = validate_one_epoch(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New Best Model Saved! QWK: {best_qwk:.4f}")

    print("Training complete.")

    # 5. Final Validation and Metric
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found, using current weights.")

    # Get raw predictions on validation set for analysis
    raw_preds, labels = get_raw_predictions(model, val_loader, device)

    # Calculate final metric using the utility (handles rounding/clipping)
    final_metric = quadratic_weighted_kappa(labels, raw_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # We need the dataframe to link back to file paths
    df_val = val_loader.dataset.df
    perform_failure_analysis(df_val, raw_preds, labels)

    # 7. Submission
    THRESHOLD = 0.8997
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        # predict_test returns processed integers [0-4]
        test_preds = predict_test(model, test_loader, device)

        # Create submission DataFrame
        df_test = test_loader.dataset.df
        submission = pd.DataFrame(
            {"id_code": df_test["id_code"], "diagnosis": test_preds}
        )

        # Save to ./submission/submission.csv
        submission.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric:.4f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
