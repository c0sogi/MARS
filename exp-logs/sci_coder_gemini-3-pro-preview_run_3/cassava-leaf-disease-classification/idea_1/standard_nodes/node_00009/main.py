import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.trainer import run_training
from library.inference import run_inference
from library.model import CassavaModel
from library.dataset import get_dataset


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Training
    # We limit epochs to Config.NUM_EPOCHS to ensure quick execution while maintaining reasonable performance.
    # The A100 GPU can handle the full dataset (15k images) very quickly.
    print("--- Starting Training ---")
    run_training(debug=False, epochs=Config.NUM_EPOCHS)

    # 3. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load validation dataset
    val_dataset = get_dataset("val", debug=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the trained model
    # pretrained=False because we are loading our own checkpoint, avoiding unnecessary downloads
    model = CassavaModel(pretrained=False, num_classes=Config.NUM_CLASSES)

    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
        )

    state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Storage for metrics and analysis
    correct_count = 0
    total_count = 0

    all_errors = []  # 1 if error, 0 if correct
    img_means = []  # Mean pixel value of input image
    img_stds = []  # Std dev of input image

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Inference
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Accuracy calculation
            batch_correct = (preds == labels).sum().item()
            correct_count += batch_correct
            total_count += labels.size(0)

            # Failure Analysis Data Collection
            # Calculate error status (0 for correct, 1 for incorrect)
            batch_errors = (preds != labels).cpu().numpy().astype(int)
            all_errors.extend(batch_errors)

            # Calculate image statistics (Mean and Std per image)
            # Flatten channel and spatial dimensions for stats: (B, C, H, W) -> (B, -1)
            flat_images = images.view(images.size(0), -1)
            batch_means = flat_images.mean(dim=1).cpu().numpy()
            batch_stds = flat_images.std(dim=1).cpu().numpy()

            img_means.extend(batch_means)
            img_stds.extend(batch_stds)

    # Calculate Final Metric
    final_acc = correct_count / total_count
    print(f"Final Validation Metric: {final_acc}")

    # Calculate Correlations
    # We use numpy to calculate Pearson correlation coefficient
    all_errors = np.array(all_errors)
    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Avoid division by zero if variance is 0
    if np.std(all_errors) > 1e-9 and np.std(img_means) > 1e-9:
        corr_mean = np.corrcoef(all_errors, img_means)[0, 1]
    else:
        corr_mean = 0.0

    if np.std(all_errors) > 1e-9 and np.std(img_stds) > 1e-9:
        corr_std = np.corrcoef(all_errors, img_stds)[0, 1]
    else:
        corr_std = 0.0

    print(f"Correlation between Error and Image Mean Intensity: {corr_mean:.4f}")
    print(f"Correlation between Error and Image Contrast (Std): {corr_std:.4f}")

    # 4. Inference on Test Set
    # Only run inference if validation metric improves upon the baseline (0.846194)
    if final_acc > 0.8461949265687584:
        print("\n--- Starting Test Inference ---")
        run_inference(debug=False)
    else:
        print(
            f"\nFinal metric {final_acc:.4f} did not beat baseline 0.8462. Skipping inference."
        )


if __name__ == "__main__":
    main()
