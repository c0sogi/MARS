import sys
import os
import torch
import numpy as np
import scipy.stats
import nltk

# -----------------------------------------------------------------------------
# Configuration Override for Fast Baseline
# -----------------------------------------------------------------------------
# We patch the configuration to limit the number of epochs for a quick run.
import library.config as config
import library.train

# Set epochs to 15 for a fast baseline execution
FAST_RUN_EPOCHS = 15
config.NUM_EPOCHS = FAST_RUN_EPOCHS
library.train.NUM_EPOCHS = FAST_RUN_EPOCHS

# -----------------------------------------------------------------------------
# Imports from Library
# -----------------------------------------------------------------------------
from library.utils import set_seed
from library.train import Trainer
from library.inference import generate_submission, Predictor
from library.data_loader import get_dataloaders


def main():
    # Set fixed seed for reproducibility
    set_seed(config.SEED)

    print(f"Starting execution with {FAST_RUN_EPOCHS} epochs...")

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    trainer.fit()

    # -------------------------------------------------------------------------
    # 2. Validation & Failure Analysis Phase
    # -------------------------------------------------------------------------
    print("Performing Final Validation and Failure Analysis...")

    # Load the best model checkpoint
    # Predictor automatically loads 'checkpoints/best_model.pth'
    predictor = Predictor()

    # Get Validation DataLoader
    # We use the same batch size as configured
    _, val_loader, _ = get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    all_preds = []
    all_targets = []
    all_lengths = []
    all_audio_energy = []
    all_errors = []

    device = predictor.device
    predictor.model.eval()

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            cls_targets = batch["cls_labels"].to(device)

            # Forward pass
            stage_outputs = predictor.model(features, mask, lengths)

            # Use the output from the final stage (Stage 3)
            final_stage = stage_outputs[-1]
            cls_probs = final_stage["cls"]

            # Decode predictions and targets using helper methods from Trainer
            batch_preds = trainer.decode_predictions(cls_probs, mask)
            batch_targets = trainer.decode_targets(cls_targets, mask)

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)
            all_lengths.extend(lengths.cpu().numpy())

            # Feature Extraction for Failure Analysis: Mean Audio Energy
            # Audio features are the last 13 channels of the input
            # features shape: (B, T, D)
            audio_feats = features[:, :, -13:]
            # Calculate mean absolute energy per valid sequence
            # Sum abs values over time (masked), divide by valid length
            valid_counts = mask.sum(dim=1).clamp(min=1)
            audio_energy = (audio_feats.abs().mean(dim=2) * mask).sum(
                dim=1
            ) / valid_counts
            all_audio_energy.extend(audio_energy.cpu().numpy())

            # Calculate sample-wise Levenshtein error
            for p, t in zip(batch_preds, batch_targets):
                dist = nltk.edit_distance(p, t)
                all_errors.append(dist)

    # Calculate Global Metric
    total_dist = sum(all_errors)
    total_len = sum(len(t) for t in all_targets)
    final_metric = total_dist / total_len if total_len > 0 else 0.0

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    print("\n--- Failure Analysis ---")
    if len(all_errors) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, _ = scipy.stats.pearsonr(all_errors, all_lengths)
        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")

        # Correlation: Error vs Audio Energy
        corr_audio, _ = scipy.stats.pearsonr(all_errors, all_audio_energy)
        print(f"Correlation (Error vs Audio Energy): {corr_audio:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # -------------------------------------------------------------------------
    # 3. Submission Phase
    # -------------------------------------------------------------------------
    threshold = 0.06789606035205364

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} meets threshold ({threshold}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
