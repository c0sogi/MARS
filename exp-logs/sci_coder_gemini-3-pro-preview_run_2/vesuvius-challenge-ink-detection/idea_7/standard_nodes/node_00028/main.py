import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, fbeta_score
from library.dataset import InkDataset
from library.model import build_model
from library.train import run_training
from library.inference import inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # We set baseline_score to 0.0 to ensure the model is saved at least once
    # (provided it learns something > 0), allowing us to perform the required
    # analysis even if the competition baseline isn't beaten immediately.
    print("--- Starting Training ---")
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, baseline_score=0.0)

    # 3. Validation & Failure Analysis
    print("--- Starting Validation & Failure Analysis ---")
    device = torch.device(Config.DEVICE)
    model = build_model()

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "Error: Model checkpoint not found. Training may have failed to produce a valid model."
        )
        return

    model.to(device)
    model.eval()

    # Load Validation Data
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = InkDataset(val_df, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []

    # Lists for failure analysis
    sample_mean_errors = []
    sample_mean_intensities = []

    with torch.no_grad():
        for images, labels, masks, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            preds_prob = torch.sigmoid(outputs)

            # --- Metric Calculation Data ---
            preds_flat = preds_prob.view(-1)
            targets_flat = labels.view(-1)
            mask_flat = masks.view(-1).bool()

            # Filter valid pixels
            valid_preds = preds_flat[mask_flat]
            valid_targets = targets_flat[mask_flat]

            all_preds.append(valid_preds.cpu())
            all_targets.append(valid_targets.cpu())

            # --- Failure Analysis Data ---
            # Calculate Mean Absolute Error per sample (patch)
            # Error = |Pred - Label|
            # We only consider valid pixels defined by 'masks'
            abs_diff = torch.abs(preds_prob - labels) * masks

            # Sum error per sample and divide by valid pixel count per sample
            # shapes: (B, 1, H, W) -> sum over (1, H, W)
            valid_pixels_per_sample = masks.sum(dim=(1, 2, 3)) + 1e-6
            mean_error_per_sample = (
                abs_diff.sum(dim=(1, 2, 3)) / valid_pixels_per_sample
            )

            # Mean Input Intensity per sample
            # images is (B, 3, H, W). We average over channels and spatial dims.
            mean_intensity_per_sample = images.mean(dim=(1, 2, 3))

            sample_mean_errors.append(mean_error_per_sample.cpu().numpy())
            sample_mean_intensities.append(mean_intensity_per_sample.cpu().numpy())

    # Calculate Final Metric
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    final_val_score = fbeta_score(
        all_preds, all_targets, beta=0.5, threshold=Config.THRESHOLD
    )

    print(f"Final Validation Metric: {final_val_score}")

    # Calculate Correlation for Failure Analysis
    errors = np.concatenate(sample_mean_errors)
    intensities = np.concatenate(sample_mean_intensities)

    if len(errors) > 1:
        correlation, _ = stats.pearsonr(errors, intensities)
        print(
            f"Failure Analysis - Correlation between Error and Input Intensity: {correlation:.10f}"
        )
    else:
        print("Failure Analysis - Insufficient data for correlation.")

    # 4. Conditional Submission
    SUBMISSION_THRESHOLD = 0.5511069832462687

    if final_val_score > SUBMISSION_THRESHOLD:
        print(
            f"Score {final_val_score} > {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        inference()
    else:
        print(f"Score {final_val_score} <= {SUBMISSION_THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
