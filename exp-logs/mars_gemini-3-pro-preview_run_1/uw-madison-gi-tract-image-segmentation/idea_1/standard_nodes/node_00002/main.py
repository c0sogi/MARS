import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import set_seed, dice_coefficient, hausdorff_distance
from library.dataset import UWMadisonDataset
from library.model import UNetResNet18
from library.train import run_training
from library.inference import run_inference

# --- Configuration ---
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 256
EPOCHS = 5
FRACTION = 0.5  # Use 50% of data for a fast but representative baseline
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./working/idea_1/best_model.pth"
SUBMISSION_DIR = "./submission"


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Starting execution on device: {DEVICE}")

    # 2. Training Phase
    print("\n=== Phase 1: Training ===")
    # Train the model using the provided training module
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        fraction=FRACTION,
        lr=1e-3,
        patience=3,
        img_size=IMG_SIZE,
    )

    # 3. Validation & Failure Analysis Phase
    print("\n=== Phase 2: Validation & Failure Analysis ===")

    # Initialize model and load best weights
    model = UNetResNet18(num_classes=3).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Loaded best model checkpoint for validation.")
    else:
        print("Warning: Checkpoint not found. Using untrained model.")

    model.eval()

    # Load the full validation dataset
    val_dataset = UWMadisonDataset(mode="val", fraction=1.0, img_size=IMG_SIZE)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Evaluating on {len(val_dataset)} validation samples...")

    dice_scores = []
    hd_scores = []
    combined_scores = []

    # Lists for failure analysis
    error_magnitudes = []
    feat_mask_areas = []
    feat_img_intensities = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            preds = (outputs > 0.5).float()

            # Calculate metrics per sample
            for i in range(images.size(0)):
                d_sample = 0
                h_sample = 0
                mask_area_sample = 0

                # Feature: Mean image intensity
                img_mean = images[i].mean().item()

                # Average metrics across the 3 classes
                for c in range(3):
                    y_true = masks[i, c]
                    y_pred = preds[i, c]

                    d_c = dice_coefficient(y_true, y_pred)
                    h_c = hausdorff_distance(y_true, y_pred)

                    d_sample += d_c
                    h_sample += h_c
                    mask_area_sample += y_true.sum().item()

                d_sample /= 3.0
                h_sample /= 3.0

                # Competition Metric: 0.4 * Dice + 0.6 * (1 - HD)
                # Note: HD is a distance (lower is better), so we invert it for the score.
                # We clip HD at 1.0 to ensure the score doesn't go negative if distance is large.
                h_score_component = max(0.0, 1.0 - h_sample)
                score = 0.4 * d_sample + 0.6 * h_score_component

                dice_scores.append(d_sample)
                hd_scores.append(h_sample)
                combined_scores.append(score)

                # Failure Analysis Data
                # Error magnitude = 1.0 - Score (Higher means worse performance)
                error_magnitudes.append(1.0 - score)
                feat_mask_areas.append(mask_area_sample)
                feat_img_intensities.append(img_mean)

    # Compute and Print Final Metrics
    final_metric = np.mean(combined_scores)
    final_dice = np.mean(dice_scores)
    final_hd = np.mean(hd_scores)

    print(f"Final Validation Metric: {final_metric:.10f}")
    print(
        f"Breakdown -> Mean Dice: {final_dice:.5f}, Mean Hausdorff Dist: {final_hd:.5f}"
    )

    # Failure Analysis: Correlations
    if len(error_magnitudes) > 1:
        corr_area, _ = pearsonr(error_magnitudes, feat_mask_areas)
        corr_intensity, _ = pearsonr(error_magnitudes, feat_img_intensities)

        print("\n--- Failure Analysis Correlations ---")
        print(f"Correlation (Error vs Mask Area): {corr_area:.4f}")
        print(f"Correlation (Error vs Image Intensity): {corr_intensity:.4f}")

        if abs(corr_area) > 0.2:
            print(
                "Insight: Model performance is sensitive to the size of the segmentation mask."
            )
        if abs(corr_intensity) > 0.2:
            print(
                "Insight: Model performance is sensitive to image brightness/contrast."
            )

    # 4. Inference Phase
    print("\n=== Phase 3: Inference & Submission ===")
    run_inference(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        checkpoint_path=CHECKPOINT_PATH,
        submission_dir=SUBMISSION_DIR,
    )

    print("Runfile Execution Complete.")


if __name__ == "__main__":
    main()
