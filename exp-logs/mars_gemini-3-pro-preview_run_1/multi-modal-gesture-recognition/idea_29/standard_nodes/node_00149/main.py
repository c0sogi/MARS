import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr

# Patch Config before other imports to ensure settings propagate
from library.config import Config

# Speed up training for the time limit
Config.NUM_EPOCHS = 12
Config.BATCH_SIZE = 8
# Ensure we use the GPU
Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from library.trainer import Trainer
from library.model import BS_MPII
from library.utils import (
    set_seed,
    generate_submission,
    median_filter,
    rle_decode,
    levenshtein_distance,
)
from library.data_loader import get_dataloaders


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Running on device: {Config.DEVICE}")
    print(f"Training for {Config.NUM_EPOCHS} epochs.")

    # Initialize Trainer and Fit
    trainer = Trainer()
    trainer.fit()

    # Load Best Model
    print("Loading best model for final evaluation...")
    device = torch.device(Config.DEVICE)
    model = BS_MPII().to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    # Validation and Failure Analysis
    val_loader = trainer.val_loader

    val_lev_dists = []
    val_seq_lengths = []
    val_gesture_counts = []

    total_dist = 0
    total_gestures = 0

    print("Evaluating on Validation Set...")
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            outputs = model(skeleton, audio, lengths)
            class_logits = outputs["class_logits"]

            # Greedy decoding
            preds = torch.argmax(class_logits, dim=2)  # (B, T)

            batch_size = preds.size(0)
            for i in range(batch_size):
                length = lengths[i].item()

                # Extract sequence
                p_seq = preds[i, :length].cpu().numpy()
                t_seq = labels[i, :length].cpu().numpy()

                # Post-processing
                p_smooth = median_filter(p_seq, window_size=Config.MEDIAN_FILTER_SIZE)
                p_gestures = rle_decode(p_smooth, min_length=Config.MIN_GESTURE_LENGTH)
                t_gestures = rle_decode(t_seq, min_length=1)

                # Metric
                dist = levenshtein_distance(p_gestures, t_gestures)

                total_dist += dist
                total_gestures += len(t_gestures)

                # Stats for analysis
                val_lev_dists.append(dist)
                val_seq_lengths.append(length)
                val_gesture_counts.append(len(t_gestures))

    final_metric = total_dist / total_gestures if total_gestures > 0 else 0.0
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    if len(val_lev_dists) > 1:
        # Correlation with Sequence Length
        corr_len, _ = pearsonr(val_lev_dists, val_seq_lengths)
        # Correlation with Number of Gestures
        corr_count, _ = pearsonr(val_lev_dists, val_gesture_counts)

        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_count:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # Submission Logic
    # Threshold from prompt
    THRESHOLD = 0.05697278911564626

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) is lower than threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        test_loader = trainer.test_loader
        predictions_map = {}

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(device)
                audio = batch["audio"].to(device)
                lengths = batch["lengths"].to(device)
                sample_ids = batch["sample_ids"]

                outputs = model(skeleton, audio, lengths)
                class_logits = outputs["class_logits"]
                preds = torch.argmax(class_logits, dim=2)

                batch_size = preds.size(0)
                for i in range(batch_size):
                    length = lengths[i].item()
                    p_seq = preds[i, :length].cpu().numpy()

                    p_smooth = median_filter(
                        p_seq, window_size=Config.MEDIAN_FILTER_SIZE
                    )
                    p_gestures = rle_decode(
                        p_smooth, min_length=Config.MIN_GESTURE_LENGTH
                    )

                    predictions_map[sample_ids[i]] = p_gestures

        generate_submission(predictions_map, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_metric:.6f}) is not lower than threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
