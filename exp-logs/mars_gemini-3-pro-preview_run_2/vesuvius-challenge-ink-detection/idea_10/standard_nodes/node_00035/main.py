import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, fbeta_score
from library.trainer import Trainer
from library.inference import run_inference
from library.dataset import InkDataset
from library.architecture import SegFormerMiTB4


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Training
    # The dataset is small (412 patches), so 15 epochs is very fast (approx 5-10 mins on A100).
    # We proceed with the default configuration to ensure best results.
    print("Initializing training pipeline...")
    trainer = Trainer()
    trainer.fit()

    # 3. Validation and Failure Analysis
    print("\nStarting Failure Analysis and Final Validation...")

    device = torch.device(Config.DEVICE)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Check if a model was saved (implies at least one epoch beat the internal save threshold)
    # If not, we fall back to the trainer's current model state, though performance might be low.
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}...")
        model = SegFormerMiTB4()
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No best model checkpoint found. Using current model state.")
        model = trainer.model

    model.to(device)
    model.eval()

    # Setup Validation Loader
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    val_dataset = InkDataset(val_df, mode="validation", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Metrics storage
    batch_scores = []
    batch_means = []
    batch_errors = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 1. Calculate Metric (F0.5)
            # Mask invalid pixels for accurate metric calculation
            probs_masked = probs * masks
            score = fbeta_score(probs_masked, labels, beta=0.5)
            batch_scores.append(score)

            # 2. Failure Analysis Data Collection
            # Feature: Mean Input Intensity (Global mean of the 3-channel slab)
            # images shape: (B, 3, H, W) -> mean over (1,2,3) -> (B,)
            # We average over the batch for correlation analysis
            avg_intensity = images.mean().item()
            batch_means.append(avg_intensity)

            # Error: Mean Absolute Error (MAE) on valid pixels
            # |Pred - Label| * Mask
            abs_err = torch.abs(probs - labels) * masks
            valid_pixel_count = masks.sum()

            if valid_pixel_count > 0:
                mae = abs_err.sum() / valid_pixel_count
            else:
                mae = torch.tensor(0.0, device=device)

            batch_errors.append(mae.item())

    # Compute Final Metric
    final_metric = np.mean(batch_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlation
    if len(batch_errors) > 1:
        corr, p_value = pearsonr(batch_means, batch_errors)
        print(
            f"Failure Analysis: Correlation between Input Intensity and Error: {corr:.4f} (p-value: {p_value:.4e})"
        )
        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation detected. Model performance varies with ink/papyrus density or scan brightness."
            )
        else:
            print(
                "Observation: No significant correlation with intensity. Errors are likely structural or texture-based."
            )

    # 4. Conditional Submission
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.597622633

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric {final_metric} > Threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        # Ensure memory is clean
        del model
        torch.cuda.empty_cache()

        # Run Inference
        run_inference()
    else:
        print(
            f"\nMetric {final_metric} <= Threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
