import sys
import os
import torch
import numpy as np
import scipy.stats
import nltk

# Import from the provided library modules
from library.config import Config, set_seed
from library.data_loader import get_loaders
from library.train import Trainer
from library.inference import generate_submission
from library.model import ICRCN
from library.utils import (
    decode_predictions,
    apply_median_filter,
    get_levenshtein_distance,
)


def main():
    # 1. Setup and Configuration
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Override Config for a fast baseline execution
    # Slightly increased epochs to accommodate larger model capacity
    Config.NUM_EPOCHS = 30

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # 2. Data Loading
    print("Loading datasets...")
    # load_cached_data=True uses pre-processed .npz files if available in ./working/cache
    train_loader, val_loader, test_loader, test_ids = get_loaders(load_cached_data=True)

    # 3. Training
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader)

    print("Starting Training...")
    trainer.train()

    # 4. Final Validation & Failure Analysis
    print("Performing Final Validation and Failure Analysis...")

    device = torch.device(Config.DEVICE)
    model = ICRCN().to(device)

    # Load the best model saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    val_errors = []
    val_lengths = []
    val_num_gestures = []
    total_distance = 0
    total_gestures = 0

    with torch.no_grad():
        for features, targets, lengths in val_loader:
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(features)

            # Use Refinement Stage 2 outputs for final analysis
            logits = outputs["ref2"]
            probs = torch.softmax(logits, dim=1)

            # Move to CPU for processing
            probs_np = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            for i in range(len(features)):
                length = lengths_np[i]

                # Extract valid sequence (remove padding)
                # Shape: (C, T) -> (T, C)
                sample_probs = probs_np[i, :, :length].transpose(1, 0)
                sample_target = targets_np[i, :length]

                # 1. Apply Median Filter Smoothing
                smoothed_preds = apply_median_filter(sample_probs, kernel_size=5)

                # 2. Decode to Sequence
                pred_seq = decode_predictions(smoothed_preds)
                target_seq = decode_predictions(sample_target)

                # 3. Compute Metric for this sample
                dist = get_levenshtein_distance(pred_seq, target_seq)

                # Accumulate for global metric
                total_distance += dist
                n_gestures = len(target_seq)
                total_gestures += n_gestures

                # Store for failure analysis
                val_errors.append(dist)
                val_lengths.append(length)
                val_num_gestures.append(n_gestures)

    # Compute Final Metric
    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    if len(val_errors) > 1:
        # Correlation between Error and Sequence Length
        corr_len, _ = scipy.stats.pearsonr(val_lengths, val_errors)
        print(f"Failure Analysis - Correlation (Error vs Seq Length): {corr_len:.4f}")

        # Correlation between Error and Number of Gestures
        corr_gest, _ = scipy.stats.pearsonr(val_num_gestures, val_errors)
        print(
            f"Failure Analysis - Correlation (Error vs Num Gestures): {corr_gest:.4f}"
        )
    else:
        print("Not enough validation samples for correlation analysis.")

    # 5. Submission
    # Threshold from requirements
    THRESHOLD = 0.1282225237449118

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
