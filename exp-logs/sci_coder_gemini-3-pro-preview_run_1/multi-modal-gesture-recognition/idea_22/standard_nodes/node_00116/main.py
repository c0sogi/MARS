import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.trainer import Trainer
from library.utils import (
    compute_levenshtein_score,
    levenshtein_distance,
    median_filter,
    rle_decode,
    set_seed,
)


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # We limit the number of epochs to ensure the run completes quickly.
    # The dataset is small, so 25 epochs is a good balance between speed and convergence.
    Config.NUM_EPOCHS = 25

    # Ensure full reproducibility
    set_seed()

    # ==========================================
    # 2. Training Phase
    # ==========================================
    # Initialize the Trainer which sets up the Model, DataLoaders, and Optimizer
    trainer = Trainer()

    # Execute the training loop
    trainer.run()

    # ==========================================
    # 3. Final Validation & Failure Analysis
    # ==========================================
    print("Running final validation and failure analysis...")

    # Load the best model checkpoint to ensure we evaluate the optimal state
    if os.path.exists(Config.BEST_MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
        )

    # Set model to evaluation mode (disables dropout, etc.)
    trainer.model.eval()

    val_preds = []
    val_targets = []

    # List to store per-sample statistics for failure analysis
    stats_data = []

    # Disable gradient computation for inference speed
    with torch.no_grad():
        for batch in trainer.val_loader:
            if batch is None:
                continue

            # Move data to the active device (GPU if available)
            skeleton = batch["skeleton"].to(trainer.device)
            audio = batch["audio"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            lengths = batch["lengths"].to(trainer.device)
            ids = batch["ids"]

            # Forward pass
            logits = trainer.model(skeleton, audio, lengths)

            # Get frame-wise predictions
            preds_raw = torch.argmax(logits, dim=2).cpu().numpy()
            targets_raw = labels.cpu().numpy()

            # Process each sample in the batch
            for i in range(len(ids)):
                length = lengths[i].item()

                # Extract valid sequence length
                p_seq = preds_raw[i, :length]
                t_seq = targets_raw[i, :length]

                # Post-processing: Median Filter to smooth noise
                p_smoothed = median_filter(
                    p_seq, window_size=Config.MEDIAN_FILTER_WINDOW
                )

                # Decode: Run-Length Encoding to get gesture list
                p_decoded = rle_decode(
                    p_smoothed,
                    background_id=Config.BACKGROUND_CLASS_ID,
                    min_length=Config.MIN_SEGMENT_LENGTH,
                )

                # Decode Targets: Ground truth sequence
                t_decoded = rle_decode(
                    t_seq,
                    background_id=Config.BACKGROUND_CLASS_ID,
                    min_length=1,  # Minimal filtering for ground truth
                )

                # Compute Levenshtein Distance for this specific sample
                dist = levenshtein_distance(p_decoded, t_decoded)

                # Append to global lists for aggregate metric
                val_preds.append(p_decoded)
                val_targets.append(t_decoded)

                # Collect stats for failure analysis
                stats_data.append(
                    {
                        "id": ids[i],
                        "seq_len_frames": length,
                        "num_gestures": len(t_decoded),
                        "levenshtein_dist": dist,
                    }
                )

    # Compute and print the Final Validation Metric
    final_metric = compute_levenshtein_score(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    if stats_data:
        df = pd.DataFrame(stats_data)

        # Calculate correlation between Error and Sequence Length
        if df["seq_len_frames"].std() > 0 and df["levenshtein_dist"].std() > 0:
            corr_len, _ = pearsonr(df["seq_len_frames"], df["levenshtein_dist"])
        else:
            corr_len = 0.0

        # Calculate correlation between Error and Number of Gestures (Complexity)
        if df["num_gestures"].std() > 0 and df["levenshtein_dist"].std() > 0:
            corr_gest, _ = pearsonr(df["num_gestures"], df["levenshtein_dist"])
        else:
            corr_gest = 0.0

        print("Failure Analysis Correlations:")
        print(f"Error vs Sequence Length: {corr_len:.4f}")
        print(f"Error vs Num Gestures: {corr_gest:.4f}")

    # ==========================================
    # 4. Submission
    # ==========================================
    # Threshold defined in the task description
    THRESHOLD = 0.05697278911564626

    if final_metric < THRESHOLD:
        trainer.generate_submission()
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
