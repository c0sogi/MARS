import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image

# Import provided library components
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_fold_dataloaders
from library.engine import train_fold, generate_submission
from library.model import EfficientNetClassifier


def run():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Adjust Config for a fast baseline run while maintaining performance
    Config.EPOCHS = 5
    # Config.N_FOLDS is already 5

    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(
        f"Starting execution with {Config.N_FOLDS} folds and {Config.EPOCHS} epochs per fold."
    )

    # Containers for Out-Of-Fold (OOF) data
    oof_preds = []
    oof_targets = []
    oof_filepaths = []

    # =========================================================================
    # 2. K-Fold Training & OOF Inference
    # =========================================================================
    for fold in range(Config.N_FOLDS):
        # --- Train ---
        # We pass epochs explicitly because the default arg in train_fold
        # was bound at import time with the original Config value.
        train_fold(fold, epochs=Config.EPOCHS)

        # --- Inference on Validation Set (OOF) ---
        print(f"Generating OOF predictions for Fold {fold}...")

        # Get validation loader for the current fold
        _, val_loader = get_fold_dataloaders(fold)

        # Load the best model for this fold
        model = EfficientNetClassifier()
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}.pth")
        load_checkpoint(checkpoint_path, model, device=device)
        model.to(device)
        model.eval()

        fold_probs = []
        fold_labels = []

        # Inference loop (No Grad for speed/memory)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                # Forward pass
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                fold_probs.extend(probs)
                fold_labels.extend(labels.numpy().flatten())

        # Store results
        oof_preds.extend(fold_probs)
        oof_targets.extend(fold_labels)

        # track filepaths for failure analysis
        # val_loader.dataset is the DogCatDataset, which holds the DataFrame
        fold_filepaths = val_loader.dataset.df["filepath"].values
        oof_filepaths.extend(fold_filepaths)

    # =========================================================================
    # 3. Global Validation Metric
    # =========================================================================
    y_true = np.array(oof_targets)
    y_pred = np.array(oof_preds)

    # Clip predictions to avoid log(0) errors, standard practice for Log Loss
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - y_pred)

    # Collect image meta-features (Width, Height, Aspect Ratio)
    print("Extracting image features from validation set...")
    widths = []
    heights = []
    aspect_ratios = []

    for fp in oof_filepaths:
        full_path = os.path.join(Config.INPUT_DIR, fp)
        try:
            # Open lazily to just get size
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except Exception as e:
            # Fallback for safety, though dataset is clean
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Calculate correlations
    # Filter out invalid images if any
    mask = widths > 0
    if np.sum(mask) > 0:
        corr_w, _ = pearsonr(errors[mask], widths[mask])
        corr_h, _ = pearsonr(errors[mask], heights[mask])
        corr_ar, _ = pearsonr(errors[mask], aspect_ratios[mask])

        print(f"Correlation (Error vs Width): {corr_w:.6f}")
        print(f"Correlation (Error vs Height): {corr_h:.6f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.6f}")
    else:
        print("Skipping correlation analysis: No valid image data found.")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    THRESHOLD = 0.02531710959157205

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not beat threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    run()
