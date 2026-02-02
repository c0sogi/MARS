import os
import sys
import torch
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.model import DSDN_GN
from library.data import get_loaders
from library.engine import (
    train_one_epoch,
    evaluate,
    reconstruct_fragments,
    predict_and_submit,
)


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between prediction error and input pixel intensity.
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    # Reconstruct predictions (probabilities)
    # We disable TTA here for speed in analysis, or keep it consistent with eval.
    # Config.TTA_ENABLED is used in evaluate, so we use it here for consistency.
    preds_map = reconstruct_fragments(model, val_loader, device, tta=Config.TTA_ENABLED)
    dataset = val_loader.dataset

    all_errors = []
    all_intensities = []

    for frag_id, pred_img in preds_map.items():
        if frag_id not in dataset.data_map:
            continue

        data = dataset.data_map[frag_id]
        target_img = data["label"]
        mask_img = data["mask"]
        volume = data["volume"]  # Tensor (65, H, W)

        if target_img is None:
            continue

        # Calculate mean intensity across Z-slices for the volume
        # Volume is normalized, but relative intensity still holds info
        # Move to numpy
        mean_intensity = torch.mean(volume, dim=0).numpy()

        # Select valid pixels
        valid_mask = mask_img > 0

        flat_preds = pred_img[valid_mask]
        flat_targets = target_img[valid_mask]
        flat_intensity = mean_intensity[valid_mask]

        # Calculate Absolute Error
        # pred_img is probability [0,1], target is binary {0,1}
        errors = np.abs(flat_preds - flat_targets)

        all_errors.append(errors)
        all_intensities.append(flat_intensity)

    if not all_errors:
        print("No validation data found for analysis.")
        return

    all_errors = np.concatenate(all_errors)
    all_intensities = np.concatenate(all_intensities)

    # Calculate Pearson Correlation
    # We check if error correlates with pixel intensity (e.g. are bright/dark areas harder?)
    if len(all_errors) > 1 and np.std(all_intensities) > 0:
        corr, _ = pearsonr(all_errors, all_intensities)
        print(
            f"Correlation between Error Magnitude and Input Pixel Intensity: {corr:.6f}"
        )
    else:
        print("Could not calculate correlation (insufficient variance or data).")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    # Using cached data for speed
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = DSDN_GN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_score = -1.0
    best_threshold = 0.5
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score, val_thresh = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val F0.5: {val_score:.4f} | Thresh: {val_thresh:.2f}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            best_threshold = val_thresh
            patience_counter = 0
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best score! Model saved to {Config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))

    # Re-evaluate to confirm metric and ensure correct state
    final_score, final_thresh = evaluate(model, val_loader, device)

    # REQUIRED: Print Final Metric
    print(f"Final Validation Metric: {final_score}")

    # Run Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 6. Submission
    # Threshold check as per requirements
    THRESHOLD_SCORE = 0.39266693592071533

    if final_score > THRESHOLD_SCORE:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device, final_thresh)
    else:
        print(
            f"\nValidation score ({final_score}) does not exceed threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    main()
