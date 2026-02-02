import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import pearsonr
import nltk

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.train import Trainer


def analyze_failures(trainer):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample Levenshtein distance and correlates it with
    sequence length and number of gestures.
    """
    trainer.model.eval()
    device = trainer.device

    all_errors = []
    all_lengths = []
    all_num_gestures = []

    print("\n--- Failure Analysis ---")

    with torch.no_grad():
        for batch in trainer.val_loader:
            if batch is None:
                continue

            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            target_sequences = batch["target_sequence"]

            # Forward Pass
            outputs = trainer.model(features, mask, lengths)

            # Use Stage 3 output
            probs = outputs["stage3"].cpu().numpy()
            lengths_cpu = lengths.cpu().numpy()

            for i in range(len(probs)):
                # 1. Get Prediction
                length = lengths_cpu[i]
                p = probs[i, :, :length].T
                frame_preds = np.argmax(p, axis=1)

                # Smoothing
                smoothed_preds = median_filter(
                    frame_preds, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
                )

                # Decode
                decoded_seq = []
                prev = -1
                for label in smoothed_preds:
                    if label != prev:
                        if label != 0:
                            decoded_seq.append(int(label))
                        prev = label

                # 2. Get Target
                target_seq = target_sequences[i]

                # 3. Compute Metric
                # Levenshtein distance for this specific sample
                dist = nltk.edit_distance(decoded_seq, target_seq)

                # Normalize by target length if possible, else raw distance
                # The global metric is sum(dist) / sum(len(target)), but for correlation
                # raw distance or normalized distance can be used.
                # We'll use raw distance as "Error Magnitude".

                all_errors.append(dist)
                all_lengths.append(length)
                all_num_gestures.append(len(target_seq))

    # Convert to arrays
    errors = np.array(all_errors)
    lengths = np.array(all_lengths)
    num_gestures = np.array(all_num_gestures)

    # Compute Correlations
    if len(errors) > 1:
        corr_len, _ = pearsonr(errors, lengths)
        corr_num, _ = pearsonr(errors, num_gestures)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Additional Stats
        print(f"Mean Error per Sequence: {np.mean(errors):.4f}")
        print(f"Max Error in a Sequence: {np.max(errors)}")
    else:
        print("Not enough samples for correlation analysis.")


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    # We reduce epochs to ensure completion within 2 hours while keeping full data
    Config.NUM_EPOCHS = 50
    set_seed(Config.SEED)

    print(f"Initializing Trainer (Epochs={Config.NUM_EPOCHS})...")

    # 2. Train
    # debug=False ensures we use the full dataset for a valid baseline
    trainer = Trainer(debug=False)
    trainer.fit()

    # 3. Load Best Model for Validation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for validation...")
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: No checkpoint found. Using current model state.")

    # 4. Final Validation
    print("Running final validation on hold-out set...")
    val_loss, val_error = trainer.validate()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_error}")

    # 5. Failure Analysis
    analyze_failures(trainer)

    # 6. Submission Logic
    THRESHOLD = 0.1282225237449118
    if val_error < THRESHOLD:
        print(
            f"Validation metric {val_error} is better than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"Validation metric {val_error} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
