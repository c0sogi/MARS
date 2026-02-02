import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
# We need to modify config before other modules use it if they import values at module level.
# However, in Python, importing the module allows modifying its attributes.
import library.config as config
from library.data_loader import get_data_loaders
from library.trainer import Trainer
from library.utils import decode_predictions, levenshtein_distance


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup and Configuration
    # Use config parameters directly
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Training
    print("Initializing trainer...")
    trainer = Trainer(device)

    print("Starting training...")
    trainer.train(train_loader, val_loader)

    # 4. Validation and Failure Analysis
    print("Performing final validation and failure analysis...")
    trainer.model.eval()

    all_errors = []
    all_lengths = []
    all_gesture_counts = []

    total_distance = 0
    total_gestures = 0

    with torch.no_grad():
        for batch in val_loader:
            features, labels, lengths, ids = batch
            features = features.to(device)
            lengths = lengths.to(device)

            # Forward pass
            outputs = trainer.model(features, lengths)
            final_logits = outputs[-1]
            probs = torch.softmax(final_logits, dim=2)

            # Process batch
            for i in range(len(ids)):
                length = lengths[i].item()
                # Get valid probability sequence
                sample_probs = probs[i, :length, :].cpu().numpy()

                # Decode prediction
                pred_seq = decode_predictions(sample_probs)

                # Process Target
                # Labels in loader are frame-wise (including 0 for background)
                # We need to extract the sequence of gestures
                sample_target_frames = labels[i, :length].cpu().numpy()
                target_seq_raw = [x for x in sample_target_frames if x != 0]
                # Collapse repeats to get the gesture list
                target_seq = [
                    x
                    for j, x in enumerate(target_seq_raw)
                    if j == 0 or x != target_seq_raw[j - 1]
                ]

                # Compute Metric for this sample
                dist = levenshtein_distance(pred_seq, target_seq)

                # Accumulate for global metric
                total_distance += dist
                total_gestures += len(target_seq)

                # Collect data for failure analysis
                all_errors.append(dist)
                all_lengths.append(length)
                all_gesture_counts.append(len(target_seq))

    # Compute Final Metric
    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    if len(all_errors) > 1:
        corr_len, _ = pearsonr(all_errors, all_lengths)
        corr_count, _ = pearsonr(all_errors, all_gesture_counts)

        print("-" * 30)
        print("Failure Analysis:")
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Gesture Count): {corr_count:.4f}")
        print("-" * 30)
    else:
        print("Not enough validation samples for failure analysis.")

    # 5. Conditional Submission
    THRESHOLD = 0.128  # Updated threshold based on previous best
    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.4f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"Metric ({final_metric:.4f}) is not below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
