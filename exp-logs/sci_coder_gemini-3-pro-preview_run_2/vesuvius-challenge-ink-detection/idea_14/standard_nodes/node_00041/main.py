import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import InkDataset
from library.model import SiameseSegFormer
from library.train import train
from library.inference import InferenceEngine


def main():
    # --- 1. Configuration & Setup ---
    # Override Config for a fast baseline execution
    Config.EPOCHS = 8  # Limit epochs to ensure fast execution
    Config.BASELINE_SCORE = (
        0.5  # Lower threshold for saving checkpoints to ensure analysis is possible
    )

    # Update submission path to match requirements
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Setup environment
    Config.setup()
    set_seed(Config.SEED)

    # --- 2. Training ---
    # Train the model. This will save the best model to Config.CHECKPOINT_PATH
    # if it exceeds Config.BASELINE_SCORE.
    train()

    # --- 3. Validation & Failure Analysis ---
    device = Config.DEVICE

    # Initialize model architecture
    model = SiameseSegFormer(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)

    # Load the best checkpoint saved during training
    # If training failed to beat baseline, this might return (0.0, 0), but we proceed for analysis
    score, epoch = load_checkpoint(model, path=Config.CHECKPOINT_PATH, device=device)
    model.eval()

    # Load Validation Dataset
    if not os.path.exists(Config.VALID_METADATA_PATH):
        print(f"Validation metadata not found at {Config.VALID_METADATA_PATH}")
        return

    val_df = pd.read_csv(Config.VALID_METADATA_PATH)
    # Use load_cached_data=True to utilize preprocessed .npy files
    val_dataset = InkDataset(val_df, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Metrics Accumulation
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # For Failure Analysis
    sample_errors = []
    sample_intensities = []

    print("Running Validation and Failure Analysis...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move data to device
            v1 = inputs["view_1"].to(device)
            v2 = inputs["view_2"].to(device)
            v3 = inputs["view_3"].to(device)
            targets = targets.to(device)

            # Inference
            logits = model(v1, v2, v3)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # --- Global Metric Calculation ---
            tp = (preds * targets).sum().item()
            fp = (preds * (1 - targets)).sum().item()
            fn = ((1 - preds) * targets).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # --- Failure Analysis Data Collection ---
            # We correlate prediction error (1 - F0.5) with input intensity.
            # Use View 2 (Center) mean intensity as the feature.

            # Calculate intensity per sample in batch: (B, 3, H, W) -> (B,)
            b_intensities = v2.mean(dim=(1, 2, 3)).cpu().numpy()

            # Calculate error per sample
            b_preds = preds.cpu().numpy()
            b_targets = targets.cpu().numpy()

            for i in range(len(b_preds)):
                p_flat = b_preds[i].flatten()
                t_flat = b_targets[i].flatten()

                _tp = np.sum(p_flat * t_flat)
                _fp = np.sum(p_flat * (1 - t_flat))
                _fn = np.sum((1 - p_flat) * t_flat)

                beta = 0.5
                epsilon = 1e-7
                precision = _tp / (_tp + _fp + epsilon)
                recall = _tp / (_tp + _fn + epsilon)

                f05_sample = ((1 + beta**2) * precision * recall) / (
                    beta**2 * precision + recall + epsilon
                )
                error = 1.0 - f05_sample

                sample_errors.append(error)
                sample_intensities.append(b_intensities[i])

    # Calculate Final Global F0.5 Score
    beta = 0.5
    epsilon = 1e-7
    global_precision = total_tp / (total_tp + total_fp + epsilon)
    global_recall = total_tp / (total_tp + total_fn + epsilon)

    final_metric = ((1 + beta**2) * global_precision * global_recall) / (
        beta**2 * global_precision + global_recall + epsilon
    )

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlation
    if len(sample_errors) > 1:
        # Using numpy for correlation
        corr_matrix = np.corrcoef(sample_errors, sample_intensities)
        correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        print(f"Failure Analysis Correlation (Error vs Intensity): {correlation:.6f}")
    else:
        print("Failure Analysis Correlation: N/A (Insufficient samples)")

    # --- 4. Submission Generation ---
    # Strict threshold from task description
    SUBMISSION_THRESHOLD = 0.597622633

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Metric exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # InferenceEngine handles loading the checkpoint internally
        engine = InferenceEngine(checkpoint_path=Config.CHECKPOINT_PATH)
        engine.generate_submission()
    else:
        print(
            f"Metric {final_metric} did not exceed threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
